# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 22:58:46 2026

@author: Administrator
"""

#面向对象--相互关联关系

#建一个专门储存关系的类
class relationship:
    def __init__(self):
        self.couple=[]
    def make_couple(self,obj1,obj2):
        self.couple=[obj1,obj2]    #清空列表--关系解除
        print("{}和{}成为对象".format(obj1.name,obj2.name))
    def get_pater(self,obj):#得到对方名字,self代表relation_obj本身
        #print("找{}的对象".format(obj.name))
        for i in self.couple:#循环列表
            if i !=obj:#代表i为obj对象
                return i
        else:#列表没有
            print("无对象")
    def break_up(self):
        print("{}和{}分手".format(self.couple[0].name,self.couple[1].name))
        self.couple.clear()
                
#人
class person:
    def __init__(self,name,age,sex,relation):
        self.name=name
        self.age=age
        self.sex=sex
        self.relation=relation   #每人实例都储存关系对象

relation_obj=relationship()
#实例
p1=person("zs",22,"F",relation_obj)
p2=person("ls",22,"M",relation_obj)
relation_obj.make_couple(p1,p2)
print(p1.relation.get_pater(p1).name)#p1对象名字
p1.relation.break_up()
p2.relation.get_pater(p2)



#组合关系--组件相互独立，但必须依赖宿主才能运行
class Dog:
    life_val=100
    def __init__(self,name,d_type,attack_val):
        self.name=name
        self.d_type=d_type
        self.attack_val=attack_val
    def bite(self,person):
        person.life_val-=self.attack_val  #咬人动作
        print('狗{}咬了人{}一口，人掉血{},还有血量{}'.format(self.name,person.name,self.attack_val,person.life_val))
#人
class person:
    life_val=100
    def __init__(self,name,age,attack_val):
        self.name=name
        self.age=age
        #self.attack_val=attack_val
        self.weapon=weapon #直接实例化
    # def beat(self,dog):
    #     dog.life_val-=self.attack_val  #咬人动作
    #     print('人{}打了狗{}一下，狗掉血{},还有血量{}'.format(self.name,dog.name,self.attack_val,dog.life_val))

#武器--组件
class weapon:
    def dog_stick(self,obj):
        self.name="打狗棒"
        self.attack_val=55
        obj.life_val-=self.attack_val
        self.print_log(obj)
    def gun(self,obj):
        self.name="Ak47"
        self.attack_val=100
        obj.life_val-=self.attack_val
        self.print_log(obj)
        

#实体--批量生产
d1=Dog("ww","柯基",15)
d2=Dog("mm","哈士奇",20)
d3=Dog("wc","柴犬",25)
p1=person("zs",18,35)
p2=person("ls",22,37)
#print(d1,d2,d3,p1,p2)

#执行动作
d1.bite(p1)  #狗咬人
p2.beat(d3)  #人打狗
#拿抢打狗
p1.weapon.gun(d2)



#