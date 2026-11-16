from selftrap.pilot_json import accounting
from selftrap.storage import write_json


def test_local_stop_is_not_an_unknown_paid_request(tmp_path):
    write_json(tmp_path/'blocked.request.json',{})
    write_json(tmp_path/'blocked.error.json',{'billing':'not_sent'})
    write_json(tmp_path/'unknown.request.json',{})
    write_json(tmp_path/'ok.request.json',{})
    write_json(tmp_path/'ok.response.json',{'usage':{'input_tokens':4,'output_tokens':5,'cost':.001},
                                           'estimated_cost':.002,'seconds':1})
    r=accounting(tmp_path)
    assert r['request_attempts']==3 and r['not_sent_requests']==1
    assert r['responses']==1 and r['unknown_outcome_requests']==1
    assert r['api_reported_cost']==.001 and r['receipt_estimated_credits']==.002
