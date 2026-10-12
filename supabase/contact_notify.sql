-- Email notification for each new contact message (Supabase + Resend)
--
-- Setup:
--   1. I Created an account at https://resend.com, then create an API key
--      (API Keys > Create API Key, permission "Sending access").
--   2. In Supabase: Database > Extensions, enable "pg_net" (if it is not already on).
--   3. I Replace re_YOUR_API_KEY and the email address below, then run this whole file
--      in the Supabase SQL Editor.
--

create extension if not exists pg_net;

-- Store (or update) the Resend key and the notification address in Vault
do $$
begin
  if exists (select 1 from vault.secrets where name = 'resend_api_key') then
    perform vault.update_secret((select id from vault.secrets where name = 'resend_api_key'), 're_YOUR_API_KEY');
  else
    perform vault.create_secret('re_YOUR_API_KEY', 'resend_api_key');
  end if;
  if exists (select 1 from vault.secrets where name = 'contact_notify_to') then
    perform vault.update_secret((select id from vault.secrets where name = 'contact_notify_to'), 'gandahostanmah@gmail.com');
  else
    perform vault.create_secret('stangandaho@gmail.com', 'contact_notify_to');
  end if;
end $$;

create or replace function private.notify_contact_message()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  v_key text := (select decrypted_secret from vault.decrypted_secrets where name = 'resend_api_key');
  v_to  text := (select decrypted_secret from vault.decrypted_secrets where name = 'contact_notify_to');
begin
  if v_key is null or v_to is null then
    return new;  -- not configured: keep the message, skip the email
  end if;

  -- Sent in the background by pg_net: the visitor's message is saved even if this fails
  perform net.http_post(
    url := 'https://api.resend.com/emails',
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || v_key,
      'Content-Type', 'application/json'
    ),
    body := jsonb_build_object(
      'from', 'Declas Website <onboarding@resend.dev>',
      'to', jsonb_build_array(v_to),
      'reply_to', new.email,
      'subject', '[Declas] ' || coalesce(new.topic, 'Message') || ' from ' || new.name,
      'text', 'Contact on Declas' || E'\n\n'
              || 'Name:  ' || new.name || E'\n'
              || 'Email: ' || new.email || E'\n'
              || 'Topic: ' || coalesce(new.topic, '-') || E'\n'
              || 'Date:  ' || to_char(new.created_at at time zone 'UTC', 'YYYY-MM-DD HH24:MI') || ' UTC' || E'\n\n'
              || new.message || E'\n\n'
              
    )
  );
  return new;
end;
$$;

revoke all on function private.notify_contact_message() from public, anon, authenticated;

drop trigger if exists contact_message_notify on public.contact_messages;
create trigger contact_message_notify
  after insert on public.contact_messages
  for each row execute function private.notify_contact_message();

-- Troubleshooting: Resend's answers to the last calls (status 200 = sent)
--   select id, status_code, content, created from net._http_response order by created desc limit 5;
