import os
import sys
import random

def ran_vals(min, max):
    v1 = random.randint(min, max)
    v2 = random.randint(min, max)

    if v1 > v2:
        v1, v2 = v2, v1
    
    v1 = (v1 + v2)/2

    return v1, v2+0.


def emit_ca_result():
    sw_vals = ran_vals(35, 71)
    xz_vals = ran_vals(53, 95)
    qx_vals = ran_vals(42, 91)
    js_vals = ran_vals(52, 87)
    nl_vals = ran_vals(50, 83)
    qys_vals = ran_vals(24, 39)
    CHAR_ANALYSIS_RESULT = [
        {
        "elements":"思维成熟度",
        "value":sw_vals[0],
        "average":sw_vals[1],
        "range":"35-71"
        },
        {
        "elements":"心智成熟度",
        "value": xz_vals[0],
        "average": xz_vals[1],
        "range":"53-95"
        },
        {"elements":"情绪成熟度",
        "value":qx_vals[0],
        "average":qx_vals[1],
        "range":"45-91"
        },
        {
        "elements":"角色成熟度",
        "value": js_vals[0],
        "average": js_vals[1],
        "range":"52-87"
        },
        {"elements":"能力成熟度",
        "value":nl_vals[0],
        "average":nl_vals[1],
        "range":"50-83"
        },
        {
        "elements":"潜意识平衡",
        "value": qys_vals[0],
        "average": qys_vals[1],
        "range":"24-39"
        }
    ]
    return CHAR_ANALYSIS_RESULT

def emit_100_result():
    xg_vals = ran_vals(45, 90)
    sx_vals = ran_vals(55, 95)
    qg_vals = ran_vals(40, 85)

    ms_match_result = [{
        "elements": "性格匹配度",
        "source_value": xg_vals[1],
        "target_value": xg_vals[0]
        },
        {
        "elements": "思想匹配度",
        "source_value": sx_vals[1],
        "target_value": sx_vals[0]
        },
        {
        "elements": "情感度",
        "source_value": qg_vals[1],
        "target_value": qg_vals[0]
        }]
    return ms_match_result

def emit_101_result():
    xg_vals = ran_vals(45, 90)
    jt_vals = ran_vals(55, 95)
    hl_vals = ran_vals(40, 85)

    ms_match_result = [{
        "elements": "性格匹配度",
        "source_value": xg_vals[1],
        "target_value": xg_vals[0]
        },
        {
        "elements": "家庭观念匹配度",
        "source_value": jt_vals[1],
        "target_value": jt_vals[0]
        },
        {
        "elements": "婚恋角色匹配度",
        "source_value": hl_vals[1],
        "target_value": hl_vals[0]
        }]
    
    return ms_match_result

def emit_102_result():
    xz_vals = ran_vals(45, 90)
    nl_vals = ran_vals(55, 95)
    mb_vals = ran_vals(40, 85)

    ms_match_result = [{
        "elements": "心智匹配度",
        "source_value": xz_vals[1],
        "target_value": xz_vals[0]
        },
        {
        "elements": "能力匹配度",
        "source_value": nl_vals[1],
        "target_value": nl_vals[0]
        },
        {
        "elements": "目标匹配度",
        "source_value": mb_vals[1],
        "target_value": mb_vals[0]
        }]
    return ms_match_result

def emit_report():
    return [
        {
        "elements": "自我认知",
        "summary": "思维成熟度，指的是看待事物有自己独立的思考，不会人云亦云。或者说拥有自己稳定的价值体系，不会轻易受他人和周围环境影响。",
        "description": "你的思维体系比较成熟. 在生活中你对人对事有自己独立的想法和判断，能够形成自己的意见，从而在生活中掌握主动权。从深层意义而言，这种思维的成熟意味着整合的价值体系。"
        },
        {
        "elements": "理性与感性",
        "summary": "一个人的心智成熟度，表现在他是如何看待自我的。他表现出来的状态与内在世界你保持真实的一致性，他们心智就达到成熟的状态。",
        "description": "你的心智比较成熟.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
        },
        {
        "elements": "情绪成熟度",
        "summary": "情绪成熟度意味着能够觉察和把握自己的情绪波动，并且能够自主地处理和表达。同时，也能够承受外界压力并恰当地做出情绪反馈的能力。",
        "description": "你的情绪比较成熟.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
        },
        {
        "elements": "角色成熟度",
        "summary": "角色成熟度指的是心理发展阶段完全匹配生理发展阶段的程度，以及其心理社会化程度对社会环境和社会规则的适应程度。",
        "description": "你的心理社会化程度比较成熟.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
        },
        {
        "elements": "能力成熟度",
        "summary": "能力成熟度意味着有足够的心理能量和技巧来面对和解决生活中出现的困境，能够清晰认知和规划自主的应对机制和目标的能力。",
        "description": "你的能力比较成熟.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
        },
                {
        "elements": "潜意识平衡",
        "summary": "潜意识平衡指的是理性世界与潜意识表达的平衡程度，代表了一个人可能存在的内心冲突的状态以及自我调和能力。",
        "description": "你的潜意识比较平衡.对于你的年龄阶段来说，你能稳定客观地看待自身挑战，你可以一步步地收获新的体验和成长."
        },
    ]