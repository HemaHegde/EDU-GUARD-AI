import { motion } from "framer-motion";
import { Shield, CheckCircle2 } from "lucide-react";

interface ConsentBannerProps {
  consented: boolean;
  onConsent: (value: boolean) => void;
}

export function ConsentBanner({ consented, onConsent }: ConsentBannerProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl border p-5 transition-colors ${
        consented
          ? "border-emerald-200 bg-emerald-50/60"
          : "border-amber-200 bg-amber-50/60"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className={`rounded-xl p-2 ${consented ? "bg-emerald-100" : "bg-amber-100"}`}>
          <Shield className={`h-5 w-5 ${consented ? "text-emerald-600" : "text-amber-600"}`} />
        </div>
        <div className="flex-1">
          <div className="text-sm font-bold text-slate-900 mb-1">
            Informed Consent — Research Data Collection
          </div>
          <div className="text-xs text-slate-600 leading-relaxed space-y-1.5">
            <p>
              This survey is part of a psychology research study on student engagement
              in online learning environments. Your responses will be used to validate
              machine learning predictions against self-reported psychological states.
            </p>
            <p>
              <strong>Data Use:</strong> Responses are stored securely and used exclusively
              for academic research purposes. All data is anonymised — no personally
              identifiable information is shared or published.
            </p>
            <p>
              <strong>Voluntary Participation:</strong> Participation is entirely voluntary.
              You may withdraw at any time without consequence.
            </p>
            <p>
              <strong>Ethics:</strong> This study follows institutional research ethics
              guidelines equivalent to IRB (Institutional Review Board) standards.
            </p>
          </div>

          <label className="mt-3 flex items-center gap-2.5 cursor-pointer group">
            <div
              className={`h-5 w-5 rounded-md border-2 flex items-center justify-center transition-all ${
                consented
                  ? "border-emerald-500 bg-emerald-500"
                  : "border-slate-300 bg-white group-hover:border-slate-400"
              }`}
              onClick={() => onConsent(!consented)}
            >
              {consented && <CheckCircle2 className="h-3.5 w-3.5 text-white" />}
            </div>
            <span
              className={`text-sm font-semibold ${consented ? "text-emerald-700" : "text-slate-700"}`}
              onClick={() => onConsent(!consented)}
            >
              I have read and consent to participate in this research study
            </span>
          </label>
        </div>
      </div>
    </motion.div>
  );
}
