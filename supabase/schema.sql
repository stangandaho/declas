-- Declas website backend (Supabase / PostgreSQL)
--
-- Setup:
--   1. Create a free project at https://supabase.com
--   2. Open SQL Editor, paste this whole file and click Run
--   3. Project Settings > API: copy the Project URL and the "anon public" key into
--      docs/assets/js/config.js (supabaseUrl, supabaseAnonKey)
--
-- Read your data in Table Editor (subscribers, downloads, contact_messages), or export
-- the subscriber list for update emails with:
--   select email, first_name, last_name, country, created_at from subscribers order by created_at;
--
-- The browser can only call the functions below; the tables themselves are not
-- reachable with the anon key. IP addresses are never stored, only a salted hash.

create extension if not exists pgcrypto with schema extensions;

create schema if not exists private;

create table if not exists private.settings (
  key text primary key,
  value text not null
);
insert into private.settings (key, value)
values ('ip_salt', encode(extensions.gen_random_bytes(32), 'hex'))
on conflict (key) do nothing;

create table if not exists public.subscribers (
  id bigint generated always as identity primary key,
  email text not null,
  first_name text not null,
  last_name text not null,
  country text not null,
  ip_hash text,
  device_id uuid,
  consent boolean not null default true,
  created_at timestamptz not null default now()
);
create unique index if not exists subscribers_email_key on public.subscribers (lower(email));
create index if not exists subscribers_ip_hash_idx on public.subscribers (ip_hash);
create index if not exists subscribers_device_idx on public.subscribers (device_id);

create table if not exists public.downloads (
  id bigint generated always as identity primary key,
  os text not null check (os in ('windows', 'macos', 'linux')),
  variant text,
  version text,
  ip_hash text,
  device_id uuid,
  created_at timestamptz not null default now()
);
create index if not exists downloads_ip_time_idx on public.downloads (ip_hash, created_at);

-- Downloads made before the website existed (shown in the total)
create table if not exists public.download_baseline (
  label text primary key,
  count integer not null check (count >= 0)
);
insert into public.download_baseline (label, count) values
  ('v1.4.0 (GitHub)', 31),
  ('v1.3.0 (GitHub)', 31),
  ('v1.1.0 (Zenodo)', 270)
on conflict (label) do nothing;

create table if not exists public.contact_messages (
  id bigint generated always as identity primary key,
  name text not null,
  email text not null,
  topic text,
  message text not null,
  ip_hash text,
  handled boolean not null default false,
  created_at timestamptz not null default now()
);

alter table public.subscribers enable row level security;
alter table public.downloads enable row level security;
alter table public.download_baseline enable row level security;
alter table public.contact_messages enable row level security;
revoke all on public.subscribers, public.downloads, public.download_baseline, public.contact_messages from anon, authenticated;
revoke all on schema private from anon, authenticated;

-- Salted hash of the caller's IP address (first entry of X-Forwarded-For)
create or replace function private.client_ip_hash()
returns text
language sql
stable
security definer
set search_path = ''
as $$
  select case when ip = '' then null else
    encode(extensions.digest(ip || (select value from private.settings where key = 'ip_salt'), 'sha256'), 'hex')
  end
  from (
    select trim(split_part(coalesce(
      current_setting('request.headers', true)::json ->> 'cf-connecting-ip',
      current_setting('request.headers', true)::json ->> 'x-forwarded-for',
      current_setting('request.headers', true)::json ->> 'x-real-ip',
      ''), ',', 1)) as ip
  ) h;
$$;

create or replace function public.download_stats()
returns json
language sql
stable
security definer
set search_path = ''
as $$
  select json_build_object(
    'total',   (select coalesce(sum(count), 0) from public.download_baseline) + (select count(*) from public.downloads),
    'windows', (select count(*) from public.downloads where os = 'windows'),
    'macos',   (select count(*) from public.downloads where os = 'macos'),
    'linux',   (select count(*) from public.downloads where os = 'linux')
  );
$$;

-- True when this browser or this IP address already filled in the form
create or replace function public.visitor_known(p_device uuid default null)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.subscribers
    where (p_device is not null and device_id = p_device)
       or (ip_hash is not null and ip_hash = private.client_ip_hash())
  );
$$;

create or replace function public.register_subscriber(
  p_email text, p_first_name text, p_last_name text, p_country text,
  p_device uuid default null, p_consent boolean default true)
returns void
language plpgsql
volatile
security definer
set search_path = ''
as $$
declare
  v_ip text := private.client_ip_hash();
begin
  p_email := lower(trim(p_email));
  if p_email !~ '^[^@\s]+@[^@\s]+\.[^@\s]{2,}$' or length(p_email) > 160 then
    raise exception 'Invalid email address';
  end if;
  if coalesce(trim(p_first_name), '') = '' or coalesce(trim(p_last_name), '') = '' or coalesce(trim(p_country), '') = '' then
    raise exception 'Missing required field';
  end if;
  if (select count(*) from public.subscribers where ip_hash = v_ip and created_at > now() - interval '1 hour') >= 10 then
    raise exception 'Too many requests';
  end if;

  insert into public.subscribers (email, first_name, last_name, country, ip_hash, device_id, consent)
  values (p_email, left(trim(p_first_name), 120), left(trim(p_last_name), 120), left(trim(p_country), 80), v_ip, p_device, coalesce(p_consent, true))
  on conflict ((lower(email))) do update
    set first_name = excluded.first_name,
        last_name  = excluded.last_name,
        country    = excluded.country,
        ip_hash    = excluded.ip_hash,
        device_id  = coalesce(excluded.device_id, public.subscribers.device_id);
end;
$$;

-- Counts a download (max 20 per hour per IP, to keep the counter honest)
create or replace function public.record_download(
  p_os text, p_variant text default null, p_version text default null, p_device uuid default null)
returns json
language plpgsql
volatile
security definer
set search_path = ''
as $$
declare
  v_ip text := private.client_ip_hash();
begin
  if p_os not in ('windows', 'macos', 'linux') then
    raise exception 'Unknown operating system';
  end if;
  if (select count(*) from public.downloads where ip_hash = v_ip and created_at > now() - interval '1 hour') < 20 then
    insert into public.downloads (os, variant, version, ip_hash, device_id)
    values (p_os, left(p_variant, 20), left(p_version, 20), v_ip, p_device);
  end if;
  return public.download_stats();
end;
$$;

create or replace function public.submit_contact(
  p_name text, p_email text, p_topic text, p_message text, p_website text default '')
returns void
language plpgsql
volatile
security definer
set search_path = ''
as $$
declare
  v_ip text := private.client_ip_hash();
begin
  -- Honeypot: bots fill the hidden "website" field. Pretend success.
  if coalesce(p_website, '') <> '' then
    return;
  end if;
  if coalesce(trim(p_name), '') = '' or coalesce(trim(p_message), '') = ''
     or trim(p_email) !~ '^[^@\s]+@[^@\s]+\.[^@\s]{2,}$' then
    raise exception 'Missing or invalid field';
  end if;
  if (select count(*) from public.contact_messages where ip_hash = v_ip and created_at > now() - interval '1 hour') >= 5 then
    raise exception 'Too many messages, please try again later';
  end if;
  insert into public.contact_messages (name, email, topic, message, ip_hash)
  values (left(trim(p_name), 120), left(lower(trim(p_email)), 160), left(p_topic, 80), left(trim(p_message), 5000), v_ip);
end;
$$;

revoke all on function private.client_ip_hash() from public, anon, authenticated;
revoke all on function public.download_stats(), public.visitor_known(uuid),
  public.register_subscriber(text, text, text, text, uuid, boolean),
  public.record_download(text, text, text, uuid),
  public.submit_contact(text, text, text, text, text) from public;
grant execute on function public.download_stats(), public.visitor_known(uuid),
  public.register_subscriber(text, text, text, text, uuid, boolean),
  public.record_download(text, text, text, uuid),
  public.submit_contact(text, text, text, text, text) to anon, authenticated;
