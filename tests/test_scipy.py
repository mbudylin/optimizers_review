"""
Примеры из статьи
# все тесты
pytest -v
# конкретный тест:
pytest -k test_scipy_scale

"""

import numpy as np
import scipy.optimize as scopt
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction
import cvxpy as cp
import pytest


def test_scipy_scale():
    # задаем параметры E, используемые в формуле Q = Q0 * exp(E * (x - 1))
    E = np.array([-3., -1., -0.5])
    # текущие цены
    P0 = np.array([10., 10., 10.])
    # текущие продажи
    Q0 = np.array([500000., 2000000., 300000.0])
    # себестоимость
    C = np.array([9.0, 8.0, 7.0])
    # текущая выручка
    R0 = np.sum(P0 * Q0)
    # текущая маржа
    M0 = np.sum((P0 - C) * Q0)

    # выручка - целевая функция, задаем возможность "управлять" масштабом через 'scale'
    def f_obj(x, args):
        f = - args['scale'] * np.sum(Q0 * P0 * x * np.exp(E * (x - 1.)))
        return f
    obj = f_obj

    # функция для ограничения по марже, по умолчанию отмасштабиируем ограничения на текущую выручку
    def f_cons(x):
        f = np.sum(Q0 * (P0 * x - C) * np.exp(E * (x - 1.0))) / R0
        return f
    cons = [scopt.NonlinearConstraint(f_cons, lb=M0 / R0, ub=np.inf)]

    # поиск новой цены производим в диапазоне 90% - 110% от текущей цены
    x_bounds = [(0.9, 1.1)] * 3
    # стартовая точка для поиска
    x0 = [1.0] * 3

    res_nonscaled = scopt.minimize(obj, x0, bounds=x_bounds, constraints=cons,
                                   method='slsqp', args={'scale': 1.})
    res_scaled = scopt.minimize(obj, x0, bounds=x_bounds, constraints=cons,
                                method='slsqp', args={'scale': 1.0 / R0})

    f_obj_val_scaled = round(f_obj(res_scaled['x'], args={'scale': 1.0}), )
    margin_scaled = round(R0 * cons[0].fun(res_scaled['x']), 1)

    f_obj_val_nonscaled = round(f_obj(res_nonscaled['x'], args={'scale': 1.0 / R0}), )
    margin_nonscaled = round(R0 * cons[0].fun(res_nonscaled['x']), 1)

    print('Решение с масштабированием ', np.round(res_scaled['x'], 3), '\n', res_scaled['message'])
    print('Значение функции ', f_obj_val_scaled)
    print('Значение маржи', margin_scaled, ' и M0', M0)
    print('------------------------------------')
    print('Решение без масштабирования', np.round(res_nonscaled['x'], 3), '\n', res_nonscaled['message'])
    print('Значение функции ', f_obj_val_nonscaled)
    print('Значение маржи', margin_nonscaled, ' и M0', M0)

    # captured = capsys.readouterr()

    # assert "Значение функции  -1" in captured.out
    # from pdb import set_trace; set_trace()

    assert [0.9, 1.016, 1.1] == list(np.round(res_scaled['x'], 3))
    assert -29210742 == f_obj_val_scaled
    assert 5399999.4 == margin_scaled

    assert [1., 1., 1.] == list(np.round(res_nonscaled['x'], 3))
    assert -1 == f_obj_val_nonscaled
    assert 5400000.0 == margin_nonscaled


def test_scipy_highs():

    # пример с "рюкзаком" у которого типы переменных различаются
    # так как задача максимизации, не забываем ставить минус
    c = -np.array([1., 2., 3., 1.])
    A_ub = np.array([[2., 1., 3., 1.]])
    b_ub = np.array([7.5])
    # проставляем индикаторы для типа переменных, 0 - непрерывное, 1 - целое число
    var_types = [1, 1, 1, 0]
    # также указываем границы, в том числе и для целочисленных переменных
    bounds = [(0, 2), (0, 1), (0, 1), (0, 1)]
    res_milp = scopt.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=var_types, method='highs')
    res_milp



def test_pyomo_pricing():
    # пример аналогичный предыдущему
    # мини пример из постановки первой статьи

    # Количество товаров
    N = 3
    # задаем эластичности, используемые в формуле Q = Q0 * exp(E * (P/P0 - 1))
    E = np.array([-3., -1., -0.5])
    # текущие цены
    P0 = np.array([10., 10., 10.])
    # текущие продажи
    Q0 = np.array([500000., 2000000., 300000.0])
    # себестоимость
    C = np.array([9.0, 8.0, 7.0])
    # текущая выручка
    R0 = np.sum(P0 * Q0)
    # текущая маржа
    M0 = np.sum((P0 - C) * Q0)
    # диапазон поиска переменных
    bounds = [(0.9, 1.1)] * N
    # объявление объекта - модели
    model = pyo.ConcreteModel('model')
    # задаем переменные, в данном случае они все непрерывные, инициализиируем 1.0
    model.x = pyo.Var(range(N), domain=pyo.Reals, bounds=bounds, initialize=1)
    # объявление целевой функции и передача в модель
    obj_expr = sum(P0[i] * model.x[i] * Q0[i] * pyo.exp(E[i] * (model.x[i] - 1)) for i in model.x)
    model.obj = pyo.Objective(expr=obj_expr, sense=pyo.maximize)
    # объявление ограничения и передача в модель
    con_expr = sum((P0[i] * model.x[i] - C[i]) * Q0[i] * pyo.exp(E[i] * (model.x[i] - 1)) for i in model.x) >= M0
    model.con = pyo.Constraint(expr=con_expr)
    # запуск солвера ipopt для решения поставленной оптимизационной задачи
    solver = pyo.SolverFactory('ipopt')
    res = solver.solve(model)
    # получение ответа - результата решения задачи
    x_opt = [round(model.x[i].value, 3) for i in model.x]
    print(x_opt, '| obj value = ', round(model.obj(x_opt), 0), '| constr value = ', round(model.con(x_opt), 1))


    # milp пример с рюкзаком
    # пример с "рюкзаком" у которого типы переменных различаются
    # так как в pyomo не реализована возможность указывать в одном векторе значения разных типов
    # то необходимо из описывать отдельно и данные, соответственно, для удобства тоже
    c_i, c_c = np.array([1., 2., 3.]), np.array([1.])
    A_i, A_c = np.array([2., 1., 3.]), np.array([1.])
    b = 7.5
    # объявление объекта - модели
    model = pyo.ConcreteModel('model')
    # формирование переменных - отдельно целочисленные и непрерывные
    bounds_i = [(0, 2), (0, 1), (0, 1)]
    bounds_c = [(0, 1)]
    model.x_i = pyo.Var(range(3), domain=pyo.Integers, bounds=bounds_i)
    model.x_c = pyo.Var(range(1), domain=pyo.Reals, bounds=bounds_c)
    # объявление целевой функции и передача в модель для максимизации
    obj_expr = sum(c_i[i] * model.x_i[i] for i in model.x_i) +\
               sum(c_c[i] * model.x_c[i] for i in model.x_c)
    model.obj = pyo.Objective(expr=obj_expr, sense=pyo.maximize)
    # объявление
    con_expr = sum(A_i[i] * model.x_i[i] for i in model.x_i) +\
               sum(A_c[i] * model.x_c[i] for i in model.x_c) <= b
    model.con = pyo.Constraint(expr=con_expr)

    solver = pyo.SolverFactory('glpk')
    res = solver.solve(model)
    x_opt = [model.x_i[i].value for i in model.x_i] + [model.x_c[i].value for i in model.x_c]
    print(x_opt, '; obj value = ', model.obj())


def test_pyomo_gdp_gap():

    model = pyo.ConcreteModel('gdp_sample')
    model.x = pyo.Var(range(0, 3), domain=pyo.Reals, bounds=(-3., 3.))
    a = [0.0, 1.0, -2.0]
    obj_expr = sum((model.x[i] - a[i]) ** 2 for i in model.x)
    model.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)
    model.djn = Disjunction(range(3))

    d = Disjunct()
    d.c = pyo.Constraint(rule=(0, model.x[0], 1))
    for i in range(3):
        model.djn[i] = [model.x[i] <= -1.5, model.x[i] == 0, model.x[i] >= 1.5]

    pyo.TransformationFactory('gdp.bigm').apply_to(model)
    solver = pyo.SolverFactory('bonmin')
    res = solver.solve(model)
    x_opt = [round(model.x[i].value, 3) for i in model.x]
    print('solution is', x_opt)


def test_cvxpy_ecos_dcp_no():

    X1, Y1 = 0.0, 0.0
    X2, Y2 = 1.0, 2.0
    # N = V2 / V1
    K = 1.5
    # задаем одну переменную x
    X = cp.Variable(1)
    Y_ = 1.0
    # целевая функция - корни из суммы квадратов - являются выпуклыми
    objective = cp.sqrt(cp.square(X - X1) + cp.square(Y_ - Y1) ** 2) + \
                cp.sqrt(cp.square(X2 - X) + cp.square(Y2 - Y_) ** 2)
    # здесь единственное ограничение - это диапазон для x
    constraints = []
    constraints.extend([X >= 0.0, X <= X2])
    # объявляем оптимизацонную задачу, здесь берем минимизацию целевой функции
    problem = cp.Problem(cp.Minimize(objective), constraints)
    # совершаем проверку задачи на выпуклость согласно правилам DCP
    print(f"is dcp: {problem.is_dcp()}")


def test_cvxpy_ecos_dcp_yes():
    X1, Y1 = 0.0, 0.0
    X2, Y2 = 1.0, 2.0
    # N = V2 / V1
    K = 1.5
    # задаем одну переменную x
    X = cp.Variable(1)
    Y_ = 1.0
    # скорректируем целевую функцию через вызов norm
    objective = cp.norm(cp.hstack([X - X1, Y_ - Y1]), 2) + K * cp.norm(cp.hstack([X2 - X, Y2 - Y_]), 2)
    # формируем ограничения и формируем задачу
    constraints = []
    constraints.extend([X >= 0.0, X <= X2])
    problem = cp.Problem(cp.Minimize(objective), constraints)
    # проверка на выпуклость
    print(f"is dcp: {problem.is_dcp()}")
    # решаем задачу путем вызова солвера ECOS
    sol = problem.solve('ECOS')
    # извлекаем решение
    x_opt = X.value[0]
    print(f'x_opt = {round(x_opt, 3)}' '| obj_val = ', round(sol, 2))

def test_cvxpy_glpk_mi():
    # объявление переменных, отдельно целочисленных и непрерывных
    x_i = cp.Variable(3, integer=True)
    x_c = cp.Variable(1, nonneg=True)
    # коэффициенты для целевой функции и ограничений
    c = np.array([1., 2., 3., 1.])
    A = np.array([2., 1., 3., 1.])
    b = 7.5
    # максимизация функции - сумма по целочисленной и непрерывной части переменных
    obj = cp.Maximize(cp.sum(c[:3] @ x_i) + cp.sum(c[3:4] @ x_c))
    # ограничения на диапазон и на общую сумму
    cons = [
        x_i[0] <= 2, x_i[1] <= 1, x_i[2] <= 1,
        x_c <= 1,
        ((A[:3] @ x_i) + (A[3:4] @ x_c)) <= b
    ]
    # формирование задачи и ее решение
    prb = cp.Problem(obj, cons)
    sol = prb.solve(verbose=False, solver='GLPK_MI')
    x_opt = np.concatenate([x_i.value, x_c.value])
    print(x_opt, '; obj value = ', sol)

