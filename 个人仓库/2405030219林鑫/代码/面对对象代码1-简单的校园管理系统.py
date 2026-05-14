class Person:#人类
    def __init__(self,name,age,sex,id_card):
        self.name=name
        self.age=age
        self.id_card=id_card
        self.sex=sex
    def get_info(self):
        print(f"姓名:{self.name},年龄:{self.age},性别:{self.sex},身份证号码:{self.id_card}")

class Teacher(Person):#教师类
    def __init__(self,name,age,sex,id_card,id_teacher,college,title):
        super().__init__(name,age,sex,id_card)
        self.id_teacher=id_teacher
        self.college=college
        self.title=title
        self.courses=[]
    def teach_course(self,course):
        if course not in self.courses:
            self.courses.append(course)
            course.teacher=self
            print(f"{self.name}成为课程{course.name}的老师")
        else:
            print(f"{self.name}已在教授{course.name}")
    def give_score(self,student,course,score):
        if course in self.courses and course in student.courses:
            student.courses[course]=score
            print(f"{self.name}{student.name}{score}")
        else:
            print(f"教授未授此课/学生未选此课")

class Student(Person):#学生类
    def __init__(self,name,age,sex,id_card,id_std,major,grade,classroom):
        super().__init__(name,age,sex,id_card)
        self.id_std=id_std
        self.major=major
        self.grade=grade
        self.classroom=classroom
        self.courses={}#key=课程名城，value=成绩
        self.gpa=0

    def select_course(self,course):#学生选课
        if course not in self.courses:
            self.courses[course]=None
            course.add_student(self)
            print(f"{self.name}成功选课：{course.name}")
        else:
            print(f"已选{course.name}")
    def drop_course(self,course):#学生退课
        if course in self.courses:
            del self.courses[course]
            course.remove_student(self)
            print(f"{self.name}成功退课：{course.name}")
        else:
            print("已退课程")
    def get_result(self):#查看成绩
        result=[]
        for course,score in self.courses.items():
            status=f"成绩{score}" if score is not None else "未评分"
            result.append((course,status))
            print(result)

    def calculate_gpa(self):
        total_credit = 0
        total_point = 0
        for course, score in self.courses.items():
            if score is not None:
                total_credit += course.credit
                # 百分制转绩点
                if score >= 90:
                    point = 4.0
                elif score >= 80:
                    point = 3.0
                elif score >= 70:
                    point = 2.0
                elif score >= 60:
                    point = 1.0
                else:
                    point = 0.0
                total_point += point * course.credit
        if total_credit > 0:
            self.gpa = round(total_point / total_credit, 2)
        return self.gpa


class Class:#班级类
    def __init__(self,name,id_class,major,grade,counselor):
        self.name=name
        self.id_class=id_class
        self.major=major
        self.grade=grade
        self.counselor=counselor
        self.students=[]
    def add_student(self,student):
        if student not in self.students:
            self.students.append(student)
    def remove_student(self,student):
        if student in self.students:
            self.students.remove(student)

class Course:#课程类
    def __init__(self,name,id_course,college,credit):
        self.name=name
        self.id_course=id_course
        self.college=college
        self.credit=credit
        self.teacher=None
        self.students=[]
    def add_student(self,student):#添加学生
        if student not in self.students:
            self.students.append(student)
    def remove_student(self,student):#移除学生
        if student in self.students:
            self.students.remove(student)
    def get_info(self):#打印课程信息
        print(f"课程:{self.name}学分:{self.credit}教师:{self.teacher.name if self.teacher is not None else None}")


class Major:#专业类
    def __init__(self,name,id_major,college):
        self.name=name
        self.id_major=id_major
        self.college=college
        self.classes=[]
        self.courses=[]
    def add_course(self,course):
        if course not in self.courses:
            self.courses.append(course)
    def add_class(self,classroom):
        if classroom not in self.courses:
            self.courses.append(classroom)

class College:#学院类
    def __init__(self,name,id_college,):
        self.name=name
        self.id_college=id_college
        self.majors=[]
        self.teachers=[]
    def add_major(self,major):
        if major not in self.majors:
            self.majors.append(major)

    def add_teacher(self,teacher):
        if teacher not in self.teachers:
            self.teachers.append(teacher)

# 测试代码 - 学校管理系统
# 1. 创建学院
college = College("信息工程学院", "C001")

# 2. 创建专业
major = Major("计算机科学与技术", "M001","信息工程学院")
college.add_major(major)  # 将专业添加到学院

# 3. 创建班级
# 需要先创建一个辅导员（教师对象）
counselor = Teacher("王老师", 35, "男", "ID001", "T001", "信息工程学院", "副教授")
college.add_teacher(counselor)  # 将教师添加到学院

# 创建班级
class_ = Class("计科1班", "CL001", major, 2023, counselor)
# 注意：原代码中 Major 类的 add_class 方法有误（错误地使用了 self.courses），这里手动添加班级到专业的 classes 列表
major.classes.append(class_)  # 正确做法：直接将班级加入专业的班级列表

# 4. 创建课程
course1 = Course("Python程序设计", "CS101","信息工程学院" , 3)
course2 = Course("数据结构", "CS102", "信息工程学院", 4)
# 将课程添加到专业的培养方案
major.courses.append(course1)
major.courses.append(course2)

# 5. 创建教师并教授课程
teacher = Teacher("李教授", 42, "男", "ID002", "T002", "信息工程学院", "教授")
college.add_teacher(teacher)

# 教师教授课程
teacher.teach_course(course1)  # 李教授成为课程Python程序设计的老师
teacher.teach_course(course2)  # 李教授成为课程数据结构的老师

# 6. 创建学生
student1 = Student("张三", 20, "男", "ID1001", "S001", major, 2023, class_)
student2 = Student("李四", 19, "女", "ID1002", "S002", major, 2023, class_)

# 将学生加入班级
class_.add_student(student1)
class_.add_student(student2)

# 7. 学生选课
print("\n--- 学生选课 ---")
student1.select_course(course1)
student1.select_course(course2)
student2.select_course(course1)

# 尝试重复选课
student1.select_course(course1)  # 应提示已选

# 8. 教师打分
print("\n--- 教师打分 ---")
teacher.give_score(student1, course1, 88)
teacher.give_score(student1, course2, 92)
teacher.give_score(student2, course1, 75)

# 9. 学生查看成绩
print("\n--- 学生成绩 ---")
student1.get_result()  # 应打印 [(course1对象, '成绩88'), (course2对象, '成绩92')]
student2.get_result()  # 应打印 [(course1对象, '成绩75')]

# 10. 计算GPA
print("\n--- 计算GPA ---")
gpa1 = student1.calculate_gpa()
gpa2 = student2.calculate_gpa()
print(f"{student1.name} 的GPA: {gpa1}")
print(f"{student2.name} 的GPA: {gpa2}")

# 11. 学生退课测试
print("\n--- 学生退课 ---")
student1.drop_course(course2)  # 退掉数据结构
student1.get_result()  # 查看退课后的成绩单（只剩Python）

# 12. 打印课程信息
print("\n--- 课程信息 ---")
course1.get_info()  # 应显示教师为李教授
course2.get_info()  # 应显示教师为李教授

# 13. 验证班级学生列表
print(f"\n班级 {class_.name} 的学生：")
for s in class_.students:
    print(f"  - {s.name}")

# 14. 验证专业课程列表
print(f"\n专业 {major.name} 的课程：")
for c in major.courses:
    print(f"  - {c.name}")

# 15. 验证学院教师列表
print(f"\n学院 {college.name} 的教师：")
for t in college.teachers:
    print(f"  - {t.name} ({t.title})")