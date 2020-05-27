import os
import sys
import math

from constant import *

from scipy.stats import norm


def calculate_XinZhi(crown_area, trunk_area, type=0):
    avg_xz = (PA_Xinzhi.Male_Value, PA_Xinzhi.Femail_Value, PA_Xinzhi.Child_16_Value)[type]
    p1 = crown_area - trunk_area
    p2 = PA_Xinzhi.Max_Value - PA_Xinzhi.Min_Value
    p3 = abs((p1 - PA_Xinzhi.Min_Value) *100 / p2)
    p4 = avg_xz * 100  / p2
    p5 = f"{PA_Xinzhi.Range_Min} ~ {PA_Xinzhi.Range_Max}"

    return {
        "value": f"{p3:.3f}",
        "average": f"{p4:.3f}",
        "range": p5
    }

def calculate_SiWei(crown_area, type=0):
    avg_sw = (PA_Siwei.Male_Value, PA_Siwei.Femail_Value, PA_Siwei.Child_16_Value)[type]
    p1 = PA_Siwei.Max_Value - PA_Siwei.Min_Value
    p2 = abs((avg_sw - PA_Siwei.Min_Value) * 100 / p1)
    p3 = (avg_sw *100 / p1)
    p4 = f"{PA_Siwei.Range_Min} ~ {PA_Siwei.Range_Max}"

    return {
        "value": f"{p2:.3f}",
        "average": f"{p3:.3f}",
        "range": p4
    }


def calculate_Qingxu(trunk_width, trunk_height, type=0):
    avg_qx = (PA_Qingxu.Male_Value, PA_Qingxu.Femail_Value, PA_Qingxu.Child_16_Value)[type]
    p1 = trunk_width / trunk_height
    p2 = PA_Qingxu.Max_Value - PA_Qingxu.Min_Value
    p3 = abs((p1 - PA_Qingxu.Min_Value) * 100 /p2 )
    p4 = (avg_qx * 100 / p2)
    p5 = f"{PA_Qingxu.Range_Min} ~ {PA_Qingxu.Range_Max}" 

    return {
        "value": f"{p3:.3f}",
        "average": f"{p4:.3f}",
        "range": p5
    }


def calculate_Juese(crown_area, crown_area_1, crown_area_2, type=0):
    avg_js = (PA_Juese.Male_Value, PA_Juese.Femail_Value, PA_Juese.Child_16_Value))[type]
    if crown_area_2 == 0 or crown_area_1 == 0:
        p1 = crown_area / 6.
    else:
        p1 = crown_area_1 / crown_area_2
    p2 = crown_area/6.
    p3 = norm.pdf(p1, 1, 1) * 100 
    p4 = PA_Juese.Max_Value - PA_Juese.Min_Value
    p5 = abs(p3 - PA_Juese.Min_Value) * 100 / p4 
    p6 = avg_js *100 /p4
    p7 = f"{PA_Juese.Range_Min} ~ {PA_Juese.Range_Max}"

    return {
        "value": f"{p5:.3f}",
        "average": f"{p6:.3f}",
        "range": p7

    }


def calculate_Nengli(crown_area, crown_area_3, crown_area_4, type=0):
    avg_nl = (PA_Nengli.Male_Value, PA_Nengli.Femail_Value, PA_Nengli.Child_16_Value))[type]
    if crown_area_4 == 0 or crown_area_3 == 0:
        p1 = crown_area / 6.
    else:
        p1 = crown_area_4 / crown_area_3
    p2 = crown_area/6.
    p3 = norm.pdf(p1, 1, 1) * 100 
    p4 = PA_Nengli.Max_Value - PA_Nengli.Min_Value
    p5 = abs(p3 - PA_Nengli.Min_Value) * 100 / p4 
    p6 = avg_js *100 /p4
    p7 = f"{PA_Nengli.Range_Min} ~ {PA_Nengli.Range_Max}"

    return {
        "value": f"{p5:.3f}",
        "average": f"{p6:.3f}",
        "range": p7

    }

def calculate_Qianyishi(crown_area, root_area):
    avg_qys = (PA_Qianyishi.Male_Value, PA_Qianyishi.Femail_Value, PA_Qianyishi.Child_16_Value))[type]
    
    p1 = root_area / crown_area
    p2 = PA_Qianyishi.Max_Value - PA_Qianyishi.Min_Value
    if crown_area == 0 or root_area == 0:
        p3 = avg_qys * 100 /p2
    else:
        p3 = norm.pdf(p1, 1, 1) * 100
    p4 = norm.pdf(avg_qys, 1, 1) * 100
    p5 = f"{PA_Qianyishi.Range_Min} ~ {PA_Qianyishi.Range_Max}"

    return {
        "value": f"{p3:.3f}",
        "average": f"{p4:.3f}",
        "range": p5
    }


def label_page(ptype):
    assert 0<= ptype < 2, "page_type must be 0 or 1."
    orient_str = ("Landscape, Portrait")[ptype]
    return {"Label": f"Tree_Page_{orient_str}"}

#_self_cognition
def label_tree_size():
    pass

def label_tree_position():
    pass

# rational & sensibility
def label_tree_crown():
    pass

def label_tree_trunk():
    pass

#思维和情绪

#潜意识
def label_tree_root():
    pass

#亲子分析

#心灵匹配 (this value should save into db)
def disposition_match_value(xz_val, sw_val, qx_val):
    return xz_val * 0.3 + sw_val * 0.5 + qx_val * 0.2

def mirrage_match_value(xz_val, js_val, qys_val):
    return xz_val * 0.4 + js_val * 0.3 + qys_val * 0.3

def cal_tree_loc(ptype, tree_on_left, tree_width):
    assert 2 > ptype >= 0, ""
    p_width = (297, 210)[ptype]
    return (tree_on_left + tree_width/2) / p_width

print(calculate_XinZhi(12, 10))

