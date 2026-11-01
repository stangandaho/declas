-- Email verification (one-time code) for the download form, sent through Brevo
--
-- Setup:
--   1. In Brevo (https://www.brevo.com): add and confirm a sender address
--      (Senders, Domains & Dedicated IPs > Senders), then create an API key (SMTP & API > API Keys).
--   2. Paste this file into the Supabase SQL Editor. In the editor only (never in this file),
--      replace xkeysib-YOUR_API_KEY and you@example.com, then click Run.
--
-- Run this BEFORE publishing the website version that asks for the code.
-- Requires schema.sql. Safe to run again.

create extension if not exists pg_net;

do $$
declare
  v_secrets text[][] := array[
    ['brevo_api_key', 'xkeysib-YOUR_API_KEY'],
    ['otp_sender_email', 'you@example.com']
  ];
  i int;
begin
  for i in 1 .. array_length(v_secrets, 1) loop
    if exists (select 1 from vault.secrets where name = v_secrets[i][1]) then
      perform vault.update_secret((select id from vault.secrets where name = v_secrets[i][1]), v_secrets[i][2]);
    else
      perform vault.create_secret(v_secrets[i][2], v_secrets[i][1]);
    end if;
  end loop;
end $$;

-- Marks who confirmed their address. Rows from before this feature stay empty.
alter table public.subscribers add column if not exists email_verified_at timestamptz;

-- Details wait here until the code is confirmed. Only a hash of the code is stored.
create table if not exists private.email_codes (
  id bigint generated always as identity primary key,
  email text not null,
  first_name text not null,
  last_name text not null,
  country text not null,
  device_id uuid,
  ip_hash text,
  code_hash text not null,
  attempts integer not null default 0,
  expires_at timestamptz not null,
  created_at timestamptz not null default now()
);
create index if not exists email_codes_email_idx on private.email_codes (email, created_at desc);
create index if not exists email_codes_ip_idx on private.email_codes (ip_hash, created_at);

create or replace function private.email_code_hash(p_email text, p_code text)
returns text
language sql
stable
security definer
set search_path = ''
as $$
  select encode(extensions.digest(
    p_email || ':' || p_code || ':' || (select value from private.settings where key = 'ip_salt'), 'sha256'), 'hex');
$$;

-- Saves the details, creates a code and emails it
create or replace function public.request_email_code(
  p_email text, p_first_name text, p_last_name text, p_country text,
  p_device uuid default null, p_lang text default 'en')
returns void
language plpgsql
volatile
security definer
set search_path = ''
as $$
declare
  v_ip text := private.client_ip_hash();
  v_key text := (select decrypted_secret from vault.decrypted_secrets where name = 'brevo_api_key');
  v_from text := (select decrypted_secret from vault.decrypted_secrets where name = 'otp_sender_email');
  v_code text;
  v_fr boolean := coalesce(p_lang, 'en') = 'fr';
begin
  p_email := lower(trim(p_email));
  p_first_name := left(trim(p_first_name), 120);
  p_last_name := left(trim(p_last_name), 120);
  if p_email !~ '^[^@\s]+@[^@\s]+\.[^@\s]{2,}$' or length(p_email) > 160 then
    raise exception 'invalid_email';
  end if;
  if char_length(coalesce(p_first_name, '')) < 2 or char_length(coalesce(p_last_name, '')) < 2
     or coalesce(trim(p_country), '') = '' then
    raise exception 'invalid_details';
  end if;
  if v_key is null or v_from is null then
    raise exception 'email_not_configured';
  end if;

  delete from private.email_codes where created_at < now() - interval '1 day';

  if exists (select 1 from private.email_codes where email = p_email and created_at > now() - interval '60 seconds') then
    raise exception 'please_wait';
  end if;
  if (select count(*) from private.email_codes where email = p_email and created_at > now() - interval '1 hour') >= 5
     or (select count(*) from private.email_codes where ip_hash = v_ip and created_at > now() - interval '1 hour') >= 10 then
    raise exception 'too_many_requests';
  end if;

  v_code := lpad((('x' || encode(extensions.gen_random_bytes(4), 'hex'))::bit(32)::bigint % 1000000)::text, 6, '0');

  insert into private.email_codes (email, first_name, last_name, country, device_id, ip_hash, code_hash, expires_at)
  values (p_email, p_first_name, p_last_name, left(trim(p_country), 80), p_device, v_ip,
          private.email_code_hash(p_email, v_code), now() + interval '10 minutes');

  perform net.http_post(
    url := 'https://api.brevo.com/v3/smtp/email',
    headers := jsonb_build_object('api-key', v_key, 'accept', 'application/json', 'content-type', 'application/json'),
    body := jsonb_build_object(
      'sender', jsonb_build_object('name', 'Declas', 'email', v_from),
      'to', jsonb_build_array(jsonb_build_object('email', p_email)),
      'subject', case when v_fr then v_code || ' est votre code Declas' else v_code || ' is your Declas code' end,
      'textContent', case when v_fr
        then 'Bonjour ' || p_first_name || ',' || E'\n\n' || 'Votre code Declas : ' || v_code || E'\n\n'
             || 'Il expire dans 10 minutes. Si vous n’avez pas demandé ce code, ignorez cet e-mail.'
        else 'Hello ' || p_first_name || ',' || E'\n\n' || 'Your Declas code: ' || v_code || E'\n\n'
             || 'It expires in 10 minutes. If you did not request it, you can ignore this email.'
      end
    )
  );
end;
$$;

-- Checks the code. A correct code saves the subscriber.
-- Returns 'ok', 'invalid', 'expired' or 'too_many' (no exceptions, so failed attempts stay counted).
create or replace function public.verify_email_code(p_email text, p_code text, p_device uuid default null)
returns text
language plpgsql
volatile
security definer
set search_path = ''
as $$
declare
  v_row private.email_codes;
begin
  p_email := lower(trim(p_email));
  select * into v_row from private.email_codes where email = p_email order by created_at desc limit 1;

  if v_row.id is null or v_row.expires_at < now() then
    return 'expired';
  end if;
  if v_row.attempts >= 5 then
    return 'too_many';
  end if;
  if v_row.code_hash <> private.email_code_hash(p_email, trim(coalesce(p_code, ''))) then
    update private.email_codes set attempts = attempts + 1 where id = v_row.id;
    return 'invalid';
  end if;

  insert into public.subscribers (email, first_name, last_name, country, ip_hash, device_id, consent, email_verified_at)
  values (p_email, v_row.first_name, v_row.last_name, v_row.country, private.client_ip_hash(),
          coalesce(p_device, v_row.device_id), true, now())
  on conflict ((lower(email))) do update
    set first_name = excluded.first_name,
        last_name = excluded.last_name,
        country = excluded.country,
        ip_hash = excluded.ip_hash,
        device_id = coalesce(excluded.device_id, public.subscribers.device_id),
        email_verified_at = excluded.email_verified_at;

  delete from private.email_codes where email = p_email;
  return 'ok';
end;
$$;

revoke all on table private.email_codes from public, anon, authenticated;
revoke all on function private.email_code_hash(text, text) from public, anon, authenticated;
revoke all on function public.request_email_code(text, text, text, text, uuid, text),
  public.verify_email_code(text, text, uuid) from public, authenticated;
grant execute on function public.request_email_code(text, text, text, text, uuid, text),
  public.verify_email_code(text, text, uuid) to anon;

-- Close the old sign-up route, which saved details without checking the address
revoke all on function public.register_subscriber(text, text, text, text, uuid, boolean) from public, anon, authenticated;
