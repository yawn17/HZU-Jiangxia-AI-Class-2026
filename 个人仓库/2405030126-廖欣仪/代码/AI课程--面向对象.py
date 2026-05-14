# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:58:24 2026

@author: Administrator
"""
#面向对象--人狗大战
#多种角色--多种实体--减少重复代码，建模板--多个实体

#攻击力与品种关联
attack_vals={
    "柯基":30,
    "哈士奇":50
    }
#模板--狗
def dog(name,d_type):
    #共同属性：数据
    data={
        "name":name,
        'd_type':d_type,
        'life_val':100
        }
    if d_type in attack_vals:
        data["attack_val"]=attack_vals[d_type]
    else:
        data["attack_val"]=15
    
    #改进dog_bite()只能狗咬人，不能人咬狗
    def dog_bite(person_obj):
        person_obj['life_val']-=data['attack_val']  #咬人动作
        print('狗{}咬了人{}一口，人掉血{},还有血量{}'.format(data['name'],person_obj['name'],data['attack_val'],person_obj['life_val']))
    
    data['bite']=dog_bite  #为了从函数外部可以调用dog_bite()
    return data
#模板--人
def people(name,age):
    data={
        "name":name,
        "age":age,
        "life_val":100
        }
    if age < 18:
        data["attack_val"]=60
    else:
        data["attack_val"]=25
    def beat(dog_obj):
        dog_obj['life_val']-=data['attack_val']   #打狗动作
        print('人{}打了狗{}一下，狗掉血{},还有血量{}'.format(data['name'],dog_obj['name'],data['attack_val'],dog_obj['life_val']))
    
    data['beat']=beat
    return data


#实体--批量生产
d1=dog("ww","柯基")
d2=dog("mm","哈士奇")
d3=dog("wc","柴犬")
p1=people("zs",12)
p2=people("ls",22)
print(d1,d2,d3,p1,p2)

#执行动作
d1['bite'](p1)  #狗咬人
p2['beat'](d3)  #人打狗
