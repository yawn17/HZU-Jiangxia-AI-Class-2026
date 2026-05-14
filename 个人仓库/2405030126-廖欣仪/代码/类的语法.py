# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:43:54 2026

@author: Administrator
"""

#类的语法

#狗
class Dog:
    d_type='田园犬'   #类属性/类变量，可定义多个--公共属性（实例共有的）
    
    def __init__(self,name,age):#初始化方法，构造方法/函数；实例化时会自动执行--私有属性
        print("汪汪",name,age)
        #绑定参数值到实例上
        self.name=name
        self.age=age
    def say_hi(self):#方法/函数--第一个参数必须为self  self代表实例本身
        print("I am a dog,my type is {},my name is {},my age is {}".format(self.d_type,self.name,self.age))


#生成实例
d1=Dog("旺旺",3)
d2=Dog("旺财",2.5)
#实例.方法
d1.say_hi()
d2.say_hi()
#实例.属性
print(d1.d_type,d2.d_type)
#仅改变d2的种类属性--为d2创建一个新属性
d2.d_type="金毛"
print(d1.d_type,d2.d_type)


#人狗大战
#狗
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
        self.attack_val=attack_val
    def beat(self,dog):
        dog.life_val-=self.attack_val  #咬人动作
        print('人{}打了狗{}一下，狗掉血{},还有血量{}'.format(self.name,dog.name,self.attack_val,dog.life_val))

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