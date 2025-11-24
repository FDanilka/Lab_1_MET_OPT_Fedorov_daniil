from Lab_1_code import SimplexMethod
sm = SimplexMethod()
sm.load_problem('test_1')
answer = sm.get_solution()
print(answer)