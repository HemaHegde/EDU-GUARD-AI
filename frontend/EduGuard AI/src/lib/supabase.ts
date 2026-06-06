import { createClient } from "@supabase/supabase-js";

const supabaseUrl =
  "https://hpangwejkwlgftzimcra.supabase.co";

const supabaseAnonKey =
  "sb_publishable_7lkl7seQSZw_e5GVqztX3g_wvrTxqa6";

export const supabase = createClient(
  supabaseUrl,
  supabaseAnonKey
);