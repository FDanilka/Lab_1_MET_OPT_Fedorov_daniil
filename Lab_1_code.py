import re

class SimplexMethod:
    def __init__(self):
        self.objective_coefficients: list[int]

        self.objective_sense: str

        self.constraint_matrix: list[list[int]]

        self.constraint_senses: list[str]

        self.constraint_rhs = list[int]

        self.canonical_problem_table: list[list]

    def load_problem(self, file_path: str) -> None:

        # Паттерн для извлечения коэффициента и индекса X из строки
        pattern = re.compile(r'^([+-]?)(\d*)x(\d+)$')

        def parse_coefficient_index(s: str) -> tuple[int, int]:# Функция для извлечения коэффициента и индекса X из строки.
            m = pattern.match(s)
            sign, coefficient, index = m.groups()
            coefficient = int(coefficient) if coefficient else 1
            if sign == '-':
                coefficient = -coefficient

            return coefficient, int(index) - 1


        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() != '']

        objective_function = lines[0]

        objective_function = objective_function.split()
        objective_function.pop(-2)
        objective_sense = objective_function.pop(-1)
        for i in range(len(objective_function)):
            objective_function[i] = parse_coefficient_index(objective_function[i])
        constraints = lines[1:]

        for i in range(len(constraints)):
            constraints[i] = constraints[i].split()
        constraint_senses = []

        constraint_rhs = []

        for c in constraints:
            constraint_senses.append(c.pop(-2))
            constraint_rhs.append(int(c.pop(-1)))

        for c in constraints:
            for i in range(len(c)):
                c[i] = parse_coefficient_index(c[i])

        # Получаем число значимых переменных
        # (оно равно наибольшему индексу X, встречающемуся в ЗЛП)
        number_of_variables = max(i for _, i in objective_function) + 1
        for c in constraints:
            c_i_max = max(i for _, i in c) + 1
            number_of_variables = max(number_of_variables, c_i_max)

        # Создаём список для коэффициентов целевой функции
        # (изначально заполняем его нулями)
        objective_coefficients = [0 for _ in range(number_of_variables)]

        # Заполняем список коэффициентами целевой функции
        # ci - (coefficient, index)
        for ci in objective_function:
            objective_coefficients[ci[1]] = ci[0]

        # Создаём матрицу с коэффициентами ограничений
        # (изначально заполняем её нулями)
        constraint_matrix = []
        for _ in range(len(constraints)):
            constraint_matrix.append([0 for _ in range(number_of_variables)])

        # Заполняем матрицу с коэффициентами ограничений соответствующими значениями
        for row_i in range(len(constraints)):
            for ci in constraints[row_i]:
                constraint_matrix[row_i][ci[1]] = ci[0]

        self.objective_coefficients = objective_coefficients
        self.objective_sense = objective_sense
        self.constraint_matrix = constraint_matrix
        self.constraint_senses = constraint_senses
        self.constraint_rhs = constraint_rhs

        self.basis_indexes = [None for _ in range(len(constraint_matrix))]

        # Преобразуем параметры ЗЛП к каноническому виду
        self._convert_problem_to_canonical_form()

    def get_solution(self, precision=8): #Основная функция

        # Коэффициенты целевой функции (coefficients)
        self.c = self.canonical_problem_table[0].copy()
        self.c.pop()

        # Цель задачи (найти максимум или минимум) (objective)
        self.obj = self.canonical_problem_table[0][-1]

        # Симплекс-таблица (simplex table)
        self.st = [row.copy() for row in self.canonical_problem_table[1:]]

        # Если в системе нет полного базиса, то формируем его
        if None in self.basis_indexes:
            self._form_basis()

        # Избавляемся от отрицательных свободных коэффициентов
        for row in self.st:
            if row[-1] < 0:
                try:
                    self._get_rid_of_negative_free_coefficients()
                except:
                    return
                break

        # Дельты симплекс-таблицы для оптимизации решения
        self.deltas = [None for _ in range(len(self.st[0]) - 1)]

        # Цикл оптимизации с помощью дельт
        while True:
            # Рассчитываем дельты
            self._calculate_deltas()

            # Проверяем решение на оптимальность
            if self._checking_all_deltas(self.obj):
                break

            # Ищем разрешающий столбец
            resolution_column_j = None

            # При заданном условии "max" - ищем столбец с минимальной дельтой
            if self.obj == "max":
                resolution_column_j = self.deltas.index(min(self.deltas))

            # При заданном условии "min" - ищем столбец с максимальной дельтой
            elif self.obj == "min":
                resolution_column_j = self.deltas.index(max(self.deltas))

            # Рассчитываем симплекс-отношения Q
            q = self._calculate_simplex_relations_of_q(resolution_column_j)

            # Если нет подходящих Q, значит решения не существует
            if all(x is None for x in q):
                return

            # Ищем строку с минимальным Q (разрешающая строка)
            # (Элемент на пересечении разрешающих столбца и строки также называется разрешающим)
            row_with_min_q_i = None
            q_min = float("inf")
            for i in range(len(q)):
                if (not (q[i] is None)) and (q[i] < q_min):
                    row_with_min_q_i = i
                    q_min = q[i]

            # Приводим разрешающий элемент к единице, а все остальные элементы в разрешающем столбце к нулю
            self._dividing_row_in_simplex_table(row_with_min_q_i, self.st[row_with_min_q_i][resolution_column_j])
            self._zero_out_other_items_in_the_column(row_with_min_q_i, resolution_column_j)

            # Обновляем базис
            self.basis_indexes[row_with_min_q_i] = resolution_column_j

        # Формируем ответ в виде: [[<Значения x_1, x_2, x_3, ...>], <Значение целевой функции F>]
        answer = [[0 for _ in range(len(self.c))], None]

        # Находим значения переменных x_1, x_2, x_3, ...
        i = 0
        for bi in self.basis_indexes:
            answer[0][bi] = self.st[i][-1]
            i += 1

        # Находим значение целевой функции F
        f = 0
        for i in range(len(self.c)):
            f += self.c[i] * answer[0][i]
        answer[1] = f

        # Убираем из итогового ответа коэффициенты тех переменных,
        # которые не входили в изначальную ЗЛП
        answer[0] = answer[0][:len(self.objective_coefficients)]

        # Возвращаем округлённый ответ
        # (чтобы избежать погрешностей Python при работе с числами)
        return self._round_result(answer, precision)

    def _convert_problem_to_canonical_form(self):
        """
        Преобразует ЗЛП к каноническому виду и записывает в единую таблицу для дальнейшей работы.
        """
        # Создаём локальные копии параметров ЗЛП
        objective_coefficients = self.objective_coefficients.copy()
        objective_sense = self.objective_sense
        constraint_matrix = [row.copy() for row in self.constraint_matrix]
        constraint_senses = self.constraint_senses.copy()
        constraint_rhs = self.constraint_rhs.copy()

        # Перебираем все типы ограничений
        for i in range(len(constraint_senses)):
            # Если знак ограничения не равен "=" (т.е. равен "<=" или ">="),
            # то нужно сделать соответствующее преобразование
            if constraint_senses[i] != "=":
                # Для знака "<=" добавляем новую переменную с коэффициентом 1
                if constraint_senses[i] == "<=":
                    constraint_matrix[i].append(1)

                # Для знака ">=" домножаем всё ограничение на -1 (чтобы изменить знак на "<="),
                # а дальше также добавляем новую переменную с коэффициентом 1
                elif constraint_senses[i] == ">=":
                    for j in range(len(constraint_matrix[i])):
                        constraint_matrix[i][j] = -constraint_matrix[i][j]
                    constraint_rhs[i] = -constraint_rhs[i]
                    constraint_matrix[i].append(1)

                # Для всех остальных строк добавляем эту новую переменную с коэффициентом 0
                objective_coefficients.append(0)
                for j in range(len(constraint_senses)):
                    if j != i:
                        constraint_matrix[j].append(0)

                # Также указываем новую переменную как базисную
                # для соответствующей строки симплекс-таблицы
                self.basis_indexes[i] = len(constraint_matrix[i]) - 1

        # Единая таблица для канонической формы ЗЛП
        canonical_problem_table = []

        # Добавляем в таблицу коэффициенты целевой функции и направление оптимизации
        canonical_problem_table.append(objective_coefficients)
        canonical_problem_table[0].append(objective_sense)

        # Добавляем в таблицу коэффициенты и правую часть ограничений
        for i in range(len(constraint_matrix)):
            canonical_problem_table.append(constraint_matrix[i])
            canonical_problem_table[i + 1].append(constraint_rhs[i])

        # Сохраняем таблицу в глобальную переменную класса
        self.canonical_problem_table = canonical_problem_table

    def _get_rid_of_negative_free_coefficients(self): #Избавляется от отрицательных свободных коэффициентов b.
        while True:
            # Найдём строку с минимальной отрицательной b
            i_min = self._find_row_with_smallest_negative_b()
            # Если такой строки нет, значит все свободные коэффициенты положительны
            if i_min is None:
                return

            # Найдём в строке i_min минимальный отрицательный элемент
            j_min = self._find_minimum_negative_item_in_row(i_min)
            if j_min is None:
                raise Exception("В строке нет отрицательных значений!")

            # Делим строку на найденный элемент
            self._dividing_row_in_simplex_table(i_min, self.st[i_min][j_min])

            # Обнуляем все другие элементы в этом столбце
            self._zero_out_other_items_in_the_column(i_min, j_min)

            # Обновляем базис
            self.basis_indexes[i_min] = j_min

    def _find_row_with_smallest_negative_b(self): #Находит строку с минимальным отрицательным b.
        b_min = 0
        i_min = None
        for i in range(len(self.st)):
            if self.st[i][-1] < b_min:
                b_min = self.st[i][-1]
                i_min = i
        return i_min

    def _find_minimum_negative_item_in_row(self, i): #Находит минимальный отрицательный элемент в строке
        row = self.st[i][:-1]
        min_element = 0
        j_min = None
        for j in range(len(row)):
            if row[j] < min_element:
                min_element = row[j]
                j_min = j
        return j_min

    def _dividing_row_in_simplex_table(self, row_i, divisor): #Делит нужную строку в таблице на заданное значение
        row = self.st[row_i]
        for j in range(len(row)):
            row[j] = row[j] / divisor

    def _zero_out_other_items_in_the_column(self, i, j): #Приводит к нулю все элементы столбца кроме заданного
        for row_i in range(len(self.st)):
            if row_i == i:
                continue
            row = self.st[row_i]
            subtraction_coeff = row[j]
            for k in range(len(row)):
                row[k] = row[k] - (self.st[i][k] * subtraction_coeff)

    def _calculate_deltas(self): #Рассчет дельт для симплекс-таблицы.
        coeffs = []
        for bi in self.basis_indexes:
            coeffs.append(self.c[bi])
        for j in range(len(self.st[0]) - 1):
            delta_j = 0
            for i in range(len(self.st)):
                delta_j += self.st[i][j] * coeffs[i]
            delta_j -= self.c[j]
            self.deltas[j] = delta_j

    def _checking_all_deltas(self, obj): #Проверка оптимальности решения с помощью дельт.
        for delta in self.deltas:
            if ((delta > 0) and (obj == "min")) or ((delta < 0) and (obj == "max")):
                return False
        return True

    def _calculate_simplex_relations_of_q(self, j): #Рассчет симплекс-отношения Q.
        q = []
        for i in range(len(self.st)):
            if self.st[i][j] <= 0:
                q.append(None)
                continue
            q.append(self.st[i][-1] / self.st[i][j])
        return q

    def _form_basis(self): # Функция формирования базиса
        for i in range(len(self.basis_indexes)):
            if self.basis_indexes[i] is None:
                for j in range(len(self.st[0]) - 1):
                    if self.st[i][j] != 0:
                        self._dividing_row_in_simplex_table(i, self.st[i][j])
                        self._zero_out_other_items_in_the_column(i, j)
                        self.basis_indexes[i] = j
                        break

    def _round_result(self, result, precision):
        if isinstance(result, list):
            return [self._round_result(x, precision) for x in result]
        elif isinstance(result, float):
            return round(result, precision)
        else:
            return result