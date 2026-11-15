"""Retry only HTTP429 admission rejections, preserving the full model deadline."""

from selftrap.providers import ProviderError
from selftrap.storage import Journal


class AdmissionJournal(Journal):
    def complete(self,label,provider,model,messages,*,retries=0,**kwargs):
        for attempt in range(retries+1):
            try:
                return super().complete(f'{label}-admission{attempt}',provider,model,messages,
                                        retries=0,**kwargs)
            except ProviderError as exc:
                # No model generation has been observed for an admission
                # rejection. Every other failure, or ANY model response,
                # terminates the trial; unknown requests are never repeated.
                if exc.status_code!=429 or attempt==retries:raise
