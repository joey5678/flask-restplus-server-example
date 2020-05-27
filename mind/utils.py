import os
import sys
import math

from constant import *


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

print(calculate_XinZhi(12, 10))

