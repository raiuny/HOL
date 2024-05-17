from simpy import Environment

env = Environment()



def func():
    a = True
    while True:
        if a:
            yield env.timeout(2)
            a = False
            print("a")
        else:
            yield env.timeout(1.1)
            print(env.now, "hello")
            yield env.timeout(0.2)
if __name__ == "__main__":
    env.process(func())
    env.run(11)