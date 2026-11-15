"""Official streaming Responses API adapter; provider details stay here."""

from time import perf_counter
import httpx
from adaption import APIConnectionError, APIStatusError

from selftrap.providers import ModelResponse, ProviderError, estimated_cost
from selftrap.throttle import retry_after_seconds


class StreamingProvider:
    def __init__(self,base):self.base=base

    def complete(self,model,messages,*,tools,max_output_tokens,timeout,tool_choice='auto'):
        if model.provider!='adaption' or model.id.startswith('stack:'):
            raise ValueError('Use a direct Adaption model ID')
        if set(model.settings)-{'temperature'}:
            raise ValueError('Unsupported model setting')
        options={'model':model.id,'input':messages,'max_output_tokens':max_output_tokens,
                 'stream':True,**model.settings}
        if tools:options['tools']=tools
        if tool_choice!='auto':options['tool_choice']=tool_choice
        started=perf_counter();final=None;events=[]
        try:
            with self.base.client.responses.create(**options,timeout=timeout) as stream:
                for event in stream:
                    events.append(event.type)
                    if perf_counter()-started>timeout:raise ProviderError('Stream total deadline exceeded')
                    if event.type in {'response.completed','response.incomplete','response.failed'}:
                        final=event.response
        except APIStatusError as exc:
            error=ProviderError(f'Adaption stream HTTP {exc.status_code}',exc.status_code==429,exc.status_code)
            error.retry_after_seconds=retry_after_seconds(exc.response.headers.get('retry-after'))
            body=getattr(exc,'body',None)
            detail=body.get('error',body) if isinstance(body,dict) else {}
            code=detail.get('code') if isinstance(detail,dict) else None
            error.quota_exhausted=code in {'insufficient_quota','credit_balance_exhausted',
                'billing_hard_limit_reached','spend_limit_exceeded'}
            if error.quota_exhausted:error.retryable=False
            raise error from None
        except APIConnectionError as exc:
            raise ProviderError(f'Adaption stream connection error: {type(exc).__name__}') from None
        except httpx.TransportError as exc:
            raise ProviderError(f'Adaption stream transport error: {type(exc).__name__}') from None
        if final is None:raise ProviderError('Stream ended without a terminal response')
        raw=final.model_dump(mode='json',exclude_none=True)
        raw['_study_stream_event_counts']={t:events.count(t) for t in set(events)}
        return ModelResponse(final.output_text or '',raw.get('output',[]),raw,perf_counter()-started,
                             raw.get('usage'),estimated_cost(model,raw.get('usage')),model.unit)

    def close(self):self.base.close()
