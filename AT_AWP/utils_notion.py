import requests, json
from datetime import datetime

token = 'secret_74Bt0gRwgFumPKSZdkDgjKLXxPXFAUpGTLPcTNXgnWX'

databaseId = 'd69f5d0d93ab4dc4b9975582c0d89bbe'

headers = {
    "Authorization": "Bearer " + token,
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}
def readDatabase(databaseId, headers):
    readUrl = f"https://api.notion.com/v1/databases/{databaseId}/query"

    res = requests.request("POST", readUrl, headers=headers)
    data = res.json()
#     print(res.status_code)
#     print(res.text)

    with open('./db.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
    return data
#     with open('./db.json', 'w', encoding='utf8') as f:
#         json.dump(data, f, ensure_ascii=False)



def createPage(databaseId, headers, model_info_dict):

    createUrl = 'https://api.notion.com/v1/pages'
    ID = model_info_dict["ID"]
    Architecture = model_info_dict["Architecture"]
    Dataset = model_info_dict["Dataset"]
    Backdoor = model_info_dict["Backdoor"]
    AT = model_info_dict["AT"]
    
    newPageData = {
        "parent": { "database_id": databaseId },
        "properties": {'Architecture': {'id': '%5Cdjo',
                                        'type': 'select',
                                        'select': {'id': '6d8f5cf5-acec-446a-a9dc-bcc38f19e422',
                                        'name': Architecture,
                                        'color': 'green'}},
                       'AT': {'id': '%60tIR',
                              'type': 'select',
                              'select': {'id': 'wnZA', 'name': AT, 'color': 'green'}},
                       'Dataset': {'id': 'bRI%40',
                                   'type': 'select',
                                   'select': {'id': '5e2ed25e-4efe-41d1-8a39-994b4b77e1e4',
                                   'name': Dataset,
                                   'color': 'green'}},
                       'Backdoor': {'id': 'uVm%60',
                                    'type': 'select',
                                    'select': {'id': '438ca02b-53e1-4145-baa2-6d4ecc1e0b0b',
                                    'name': Backdoor,
                                    'color': 'green'}},
                       'ID': {'id': 'title',
                              'type': 'title',
                              'title': [{'type': 'text',
                              'text': {'content': ID, 'link': None},
                              'annotations': {'bold': False,
                              'italic': False,
                              'strikethrough': False,
                              'underline': False,
                              'code': False,
                              'color': 'default'},
                              'plain_text': ID,
                              'href': None}]}}
    }
    
    data = json.dumps(newPageData)
    # print(str(uploadData))

    res = requests.request("POST", createUrl, headers=headers, data=data)

    print(res.status_code)
    print(res.text)




def updatePage(padeId, headers, model_info_dict):
    updateUrl = f"https://api.notion.com/v1/pages/{pageId}"
    ID = model_info_dict["ID"]
    Architecture = model_info_dict["Architecture"]
    Dataset = model_info_dict["Dataset"]
    Backdoor = model_info_dict["Backdoor"]
    AT = model_info_dict["AT"]
    updateData = {
        "properties": {'Architecture': {'id': '%5Cdjo',
                                        'type': 'select',
                                        'select': {'id': '6d8f5cf5-acec-446a-a9dc-bcc38f19e422',
                                        'name': Architecture,
                                        'color': 'green'}},
                       'AT': {'id': '%60tIR',
                              'type': 'select',
                              'select': {'id': 'wnZA', 'name': AT, 'color': 'green'}},
                       'Dataset': {'id': 'bRI%40',
                                   'type': 'select',
                                   'select': {'id': '5e2ed25e-4efe-41d1-8a39-994b4b77e1e4',
                                   'name': Dataset,
                                   'color': 'green'}},
                       'Backdoor': {'id': 'uVm%60',
                                    'type': 'select',
                                    'select': {'id': '438ca02b-53e1-4145-baa2-6d4ecc1e0b0b',
                                    'name': Backdoor,
                                    'color': 'green'}},
                       'ID': {'id': 'title',
                              'type': 'title',
                              'title': [{'type': 'text',
                              'text': {'content': ID, 'link': None},
                              'annotations': {'bold': False,
                              'italic': False,
                              'strikethrough': False,
                              'underline': False,
                              'code': False,
                              'color': 'default'},
                              'plain_text': ID,
                              'href': None}]}}
    }

    data = json.dumps(updateData)

    response = requests.request("PATCH", updateUrl, headers=headers, data=data)

    print(response.status_code)
    print(response.text)