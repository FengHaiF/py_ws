#  装饰器学习:

# 1.装饰器的基础是：闭包
'''
# 定义内层函数，在内层函数访问enclosing 变量
# 返回内层函数（非必需）
# 01. easy demo
def outer( a ):
    b = 10
    # inner是内函数
    def inner():
          #在内函数中 用到了外函数的临时变量
        print(a+b)
        # 外函数的返回值是内函数的引用
    return inner

# 调用
demo = outer(5)
demo()

# 02.修改闭包变量的实例
#修改闭包变量的实例
# outer是外部函数 a和b都是外函数的临时变量
def outer( a ):
    b = 10  # a和b都是闭包变量
    c = [a] #这里对应修改闭包变量的方法2
    # inner是内函数
    def inner():
        #内函数中想修改闭包变量
        # 方法1 nonlocal关键字声明
        nonlocal b
        b+=1
        # 方法二，把闭包变量修改成可变数据类型 比如列表
        c[0] += 1
        print(c[0])
        print(b)
    # 外函数的返回值是内函数的引用
    return inner

demo = outer(5)
demo() # 6  11

# 03.使用闭包的过程中，一旦外函数被调用一次返回了内函数的
# 引用，虽然每次调用内函数，是
# 开启一个函数执行过后消亡，但是闭包变量实际上只有一份，
# 每次开启内函数都在使用同一份闭包变量
def outer(x):
    def inner(y):
        nonlocal x
        x+=y
        return x
    return inner


a = outer(10)
print(a(1)) //11
print(a(3)) //14
# 次调用inner的时候，使用的闭包变量x实际上是同一个。

'''

# 装饰器：
# 装饰器本质上是一个函数，该函数用来处理其他函数，它可以
# 让其他函数在不需要修改代码的前提下增加额外的功能，装饰
# 器的返回值也是一个函数对象。它经常用于有切面需求的场景，
# 比如：插入日志、性能测试、事务处理、缓存、权限校验等应
# 用场景。装饰器是解决这类问题的绝佳设计，有了装饰器，我
# 们就可以抽离出大量与函数功能本身无关的雷同代码并继续重
# 用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。

import time

def show_time(func):

#  测试函数运行时间
#  用到了闭包！！！

    def wrapper():
        print("Begin:",time.ctime())
        func()
        print("End:",time.ctime())

    return wrapper

@show_time # foo = show_time(foo)
def foo():
    print("I am foo...")
    time.sleep(1)

foo()
# Begin: Sun Aug 25 11:01:56 2019
# I am foo...
# End: Sun Aug 25 11:01:57 2019

