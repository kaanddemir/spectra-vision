/**
 * One action that downloads the analysis as JSON: Spectra's proposals and evidence with the reviewer's
 * decisions, adjusted times and notes. No review work is required first. The browser's own download is the
 * only confirmation; a failure is reported once, and an unreachable server is left to the global banner.
 */
import { useMutation } from "@tanstack/react-query";
import { useRef } from "react";

import { api, errorMessage, isOffline } from "../../api/client";
import { saveFile } from "../../lib/download";
import { Button } from "../../ui/Button";
import { icons } from "../../ui/icons";
import { useToast } from "../../ui/Toast";

export function ExportButton({ analysisId }: { analysisId: string }) {
  const toast = useToast();
  const sending = useRef(false);
  const mutation = useMutation({ mutationFn: () => api.exportAnalysis(analysisId) });

  const start = () => {
    if (sending.current) return; // repeated clicks before the button updates make one download
    sending.current = true;
    mutation.mutate(undefined, {
      onSuccess: ({ blob, fileName }) => saveFile(blob, fileName),
      onError: (error) => {
        if (isOffline(error)) return;
        toast({ tone: "danger", key: "export-failed", title: "Could not export", description: errorMessage(error) });
      },
      onSettled: () => {
        sending.current = false;
      },
    });
  };

  return (
    <Button icon={icons.download} loading={mutation.isPending} onClick={start}>
      {mutation.isPending ? "Exporting…" : "Export JSON"}
    </Button>
  );
}
