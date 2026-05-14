# -*- coding:utf-8 -*-
import os
import re
import json
import requests
import random
from py2neo import Graph
from NLU.Chatty_intention.clf_model import CLFModel
from kg_config import *
from kg_utils import *
graph = graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
clf_model = CLFModel('./NLU/Chatty_intention/save_model/')

def classifier(text):
    """
    判断是否是闲聊意图，以及是什么类型闲聊
    """
    return clf_model.predict(text)

def intent_classifier(text):
    url = 'http://127.0.0.1:8080/service/api/bert_intent_recognize'
    data = {"text": text}
    headers = {'Content-Type': 'application/json; charset=utf8'}
    reponse = requests.post(url, data=json.dumps(data), headers=headers)
    if reponse.status_code == 200:
        reponse = json.loads(reponse.text)
        return reponse['res']
    else:
        return -1

def slot_recognizer(sample):
    url = 'http://127.0.0.1:6002/service/api/medical_ner'
    data = {"text": sample}
    headers = {'Content-Type': 'application/json;charset=utf8'}
    reponse = requests.post(url, data=json.dumps(data), headers=headers)
    if reponse.status_code == 200:
        reponse = json.loads(reponse.text)
        return reponse['result']
    else:
        return -1

def gossip_robot(intent):
    return random.choice(
                gossip_corpus.get(intent)
            )

def medical_robot(text):
    """
    如果确定是诊断意图则使用该方法进行诊断问答
    """
    semantic_slot = semantic_parser(text)
    answer = get_answer(semantic_slot)
    return answer

def semantic_parser(text):
    intent_rst = intent_classifier(text)
    print(f'intent_rst: {intent_rst}')
    slot_rst = slot_recognizer(text)
    print(f'slot_rst: {slot_rst}')
    if intent_rst == -1 or slot_rst == -1 or len(slot_rst) == 0 or intent_rst.get("name") == "其他":
        return semantic_slot.get("unrecognized")
    slot_info = semantic_slot.get(intent_rst.get("name"))

    slots = slot_info.get("slot_list")
    slot_values = {}
    for key, value in slot_rst.items():
        if value.lower() == slots[0].lower():
            slot_values[slots[0]] = key
    slot_info["slot_values"] = slot_values

    # 根据意图强度来确认回复策略
    threhold = intent_rst.get("confidence")

    if threhold >= intent_threshold_config["accept"]:
        slot_info["intent_strategy"] = "accept"
    elif threhold >= intent_threshold_config["deny"]:
        slot_info["intent_strategy"] = "clarify"
    else:
        slot_info["intent_strategy"] = "deny"

    return slot_info

def get_answer(slot_info):
    cql_template = slot_info.get("cql_template")
    reply_template = slot_info.get("reply_template")
    ask_template = slot_info.get("ask_template")
    slot_values = slot_info.get("slot_values")
    strategy = slot_info.get("intent_strategy")
    print(f'11slot_info-->{slot_info}')
    if not slot_values:
        return slot_info

    if strategy == "accept":
        cql = []
        if isinstance(cql_template, list):
            for cqlt in cql_template:
                cql.append(cqlt.format(**slot_values))
        else:
            cql = cql_template.format(**slot_values)
        answer = neo4j_searcher(cql)
        if not answer:
            slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
        else:
            pattern = reply_template.format(**slot_values)
            slot_info["replay_answer"] = pattern + answer
    elif strategy == "clarify":
        print(f'进入clarify')
        # 澄清用户是否问该问题
        pattern = ask_template.format(**slot_values)
        slot_info["replay_answer"] = pattern
        # 得到肯定意图之后需要给用户回复的答案
        cql = []
        if isinstance(cql_template, list):
            for cqlt in cql_template:
                cql.append(cqlt.format(**slot_values))
        else:
            cql = cql_template.format(**slot_values)
        print(f'cql--》{cql}')
        answer = neo4j_searcher(cql)
        if not answer:
            slot_info["replay_answer"] = "唔~我装满知识的大脑此刻很贫瘠"
        else:
            pattern = reply_template.format(**slot_values)
            slot_info["choice_answer"] = pattern + answer
    elif strategy == "deny":
        slot_info["replay_answer"] = slot_info.get("deny_response")

    return slot_info

def neo4j_searcher(cql_list):
    result = ''
    if isinstance(cql_list, list):
        for cql in cql_list:
            rst = []
            data = graph.run(cql).data()
            if not data:
                continue
            for d in data:
                d = list(d.values())
                if isinstance(d[0], list):
                    rst.extend(d[0])
                else:
                    rst.extend(d)

            data = "、".join([str(i) for i in rst])
            result += data + "\n"
    else:
        data = graph.run(cql_list).data()
        if not data:
            return result
        rst = []
        for d in data:
            d = list(d.values())
            rst.extend(d)

        data = "、".join([str(i) for i in rst])
        result += data

    return result


if __name__ == '__main__':
    print("你好，我是你的AI医生\n")
    while True:
        user_input = input()
        # 第一次意图识别
        user_intent = classifier(user_input)
        print(f'user_intent-->{user_intent}')
        if user_intent in ["greet", "goodbye", "deny", "isbot"]:
            print(f'这是闲聊')
            reply = gossip_robot(user_intent)
            print(f'闲聊的reply--》{reply}')
        elif user_intent == "accept":
            print(f'这是accept')
            reply = load_user_dialogue_context()
            reply = reply.get("choice_answer")
            print(f'回复111--》{reply}')
        else:
            print(f'这是medical_robot')
            reply = medical_robot(user_input)
            print('@' * 80)
            print(f'medical_robot-->reply--》{reply}')
            if reply["slot_values"]:
                dump_user_dialogue_context(reply)
            reply = reply.get("replay_answer")
            print(f'回复--》{reply}')
