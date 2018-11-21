import numpy as np
import re

from collections import namedtuple


"""*********************************************************************************
                                Global Variables
*********************************************************************************"""
file = open("testcase01.txt", "r")
data = file.read().splitlines()

varNumber = int(data[0]) + 1  # We add 1 to manipulate the output variable too
testCase = data[1:]

variables = []  # Variables names
crispInput = []  # Crisp input corresponds to each variable in variables array
linguistics = []  # Contain each variable array of linguistics
linguistics_values = []  # x_coordinates of trapezoidal or triangular set corresponds to each linguistic of a variable
rules = []  # Contains Rule Tuples of premises and operands
variables_membership_tuples = {}  # This dictionary holds the structure of the named tuples
variables_membership = {}  # This array holds the membership tuple of each variable
rules_inference = []  # Array of tuples of output variable inferred value
output_var_centroids = {}  # Holds the centroid of each linguistic
inferred_output_variable = 0


"""#############################################################################################################
You Should Read About The Named Tuples In Python To Understand The Next Tuple Definition And Fuzzification Step
#############################################################################################################"""

Rule = namedtuple('Rule', ['premises', 'operators'])  # Note that premises contains the output variable conclusion
Point = namedtuple('Point', ['x_coordinate', 'y_coordinate'])


"""*********************************************************************************
                                Extracting Variables
*********************************************************************************"""
index = 0

for i in range(varNumber):

    # Splitting the variable and its crisp input
    var = testCase[index].split(" ")

    # This condition is to handle the output
    # variable as it does not has a crisp input
    if len(var) > 1:
        crispInput.append(float(var[1]))

    variables.append(var[0])

    linguistics.append([])
    linguistics_values.append([])

    # Starts at the 1st set input
    varIndex = index + 2
    numberOfSets = int(testCase[index + 1])
    for j in range(numberOfSets):

        linguistic = testCase[varIndex].split(" ")[0]
        linguistic_values = np.array(testCase[varIndex+1].split(" "), float)

        linguistics[i].append(linguistic)
        linguistics_values[i].append(linguistic_values)

        # Starts at the next set input
        varIndex += 2

    # Starts at the next variable 1st set input
    # 2: for the 1st 2 lines that define the variable and number of sets
    index += numberOfSets * 2 + 2

"""*********************************************************************************
                            Extracting Rules
*********************************************************************************"""
rulesNumber = int(testCase[index])
index += 1

for i in range(rulesNumber):
    premises = re.findall(r'\w+\s*=\s*\w+', testCase[index][2:])
    operatorsOrdered = re.findall(r'(AND|OR)', testCase[index][2:])
    rule = Rule(premises, operatorsOrdered)
    rules.append(rule)
    index += 1

"""*********************************************************************************
                                Fuzzification
*********************************************************************************"""


def membership(p1, p2, x):

    # Check for division by zero
    if p2.x_coordinate-p1.x_coordinate != 0:
        slope = (p2.y_coordinate-p1.y_coordinate)/(p2.x_coordinate-p1.x_coordinate)
    else:
        return 0

    y = slope * (x - p1.x_coordinate) + p1.y_coordinate

    return y


def intersection(x_coordinates, x):

    # If trapezoid
    y_coordinates = [0, 1, 1, 0]

    # If triangular
    if len(x_coordinates) == 3:
        y_coordinates = [0, 1, 0]

    i = 0
    # Determine the points of the line that intersects with the X=x Line
    while i < len(x_coordinates) and x > x_coordinates[i]:
        i += 1

    if i < len(x_coordinates):
        p1 = Point(x_coordinates[i - 1], y_coordinates[i - 1])
        p2 = Point(x_coordinates[i], y_coordinates[i])
    else:
        return 0

    return membership(p1, p2, x)


# Constructing the named tuples structure for
# each variable to hold the membership later on
# len(variables)-1: as the output variable is not considered in fuzzification step
for i in range(len(variables)-1):
    tuple_names = []

    for j in range(len(linguistics[i])):
        tuple_names.append(linguistics[i][j])

    Membership = namedtuple('Membership', tuple_names)
    variables_membership_tuples[variables[i]] = Membership


# Calculating the membership for each variable
# len(variables)-1: as the output variable is not considered in fuzzification step
for i in range(len(variables)-1):
    linguistics_memberships = []

    # Calculating the membership of the linguistic(i)(j) for each the variable(i)
    for j in range(len(linguistics[i])):
        linguistic_membership = intersection(linguistics_values[i][j], crispInput[i])
        linguistics_memberships.append(linguistic_membership)

    # Using the named tuple structure we defined to hold the variable membership
    variable_membership = variables_membership_tuples[variables[i]](*linguistics_memberships)
    variables_membership[variables[i]] = variable_membership

"""*********************************************************************************
                                Inference
*********************************************************************************"""


# This Function Used To Achieve Operators Precedence.
# We Can Make It More Efficient And Use Disjoint Sets Instead Of Dictionary
# As We May, At The Worst Case Iterate All Dictionary Elements.
# Simply It Updates All old_val Values With The new_val Value.
def update_dictionary(my_dict, old_val, new_val):
    if old_val == new_val:
        return my_dict

    keys = list(my_dict.keys())
    values = list(my_dict.values())

    while old_val in my_dict.values():
        key_index = values.index(old_val)
        key = keys[key_index]
        my_dict[key] = new_val
        values[key_index] = new_val

    return my_dict


def rule_inference(_rule, _operations):  # rule, rule_number,

    my_dict = {}

    for operation in _operations:

        if operation[1] in my_dict:
            operand1 = my_dict[operation[1]]
        else:
            operand1 = _rule.premises[operation[1]]
            operand1 = re.split(r'\s+=\s+', operand1)
            operand1 = getattr(variables_membership[operand1[0]], operand1[1])

        if operation[2] in my_dict:
            operand2 = my_dict[operation[2]]
        else:
            operand2 = _rule.premises[operation[2]]
            operand2 = re.split(r'\s+=\s+', operand2)
            operand2 = getattr(variables_membership[operand2[0]], operand2[1])

        if operation[0] == "AND":
            new_val = min(operand1, operand2)
        else:  # if operation[0] == "OR"
            new_val = max(operand1, operand2)

        old_val = operand2 if new_val == operand1 else operand1
        my_dict[operation[1]] = new_val
        my_dict[operation[2]] = new_val

        # Note that this update_dictionary implementation would
        # be faster if replaced with disjoint sets xD
        my_dict = update_dictionary(my_dict, old_val, new_val)

    # Note That All Of The Dictionary Keys Have The Same Value At This Step
    output_inference = re.split(r'\s+=\s+', _rule.premises[-1])[1]
    rules_inference.append((my_dict[0], output_inference))


# Preparing each rule operations
operations = []
for i in range(len(rules)):
    operations.append([])

    for j in range(len(rules[i].operators)):
        operations[i].append((rules[i].operators[j], j, j+1))

    operations[i].sort(key=lambda tup: tup[0])

# Executing operations to inference the output variables
for i in range(len(rules)):
    rule_inference(rules[i], operations[i])

"""*********************************************************************************
                                Defuzzification
*********************************************************************************"""


def signed_area(x_coordinates, y_coordinates):
    sum = 0.0
    for i in range(len(x_coordinates)-1):
        sum += x_coordinates[i]*y_coordinates[i+1] - x_coordinates[i+1]*y_coordinates[i]
    area = sum/2.0
    return area


def centroid_x(x_coordinates):

    # If trapezoid
    y_coordinates = [0, 1, 1, 0]

    # If triangular
    if len(x_coordinates) == 3:
        y_coordinates = [0, 1, 0]

    sum = 0.0
    area = signed_area(x_coordinates, y_coordinates)

    for i in range(len(x_coordinates)-1):
        sum += (x_coordinates[i] + x_coordinates[i+1])*(x_coordinates[i]*y_coordinates[i+1] - x_coordinates[i+1]*y_coordinates[i])

    centroid = sum/(6*area)

    return centroid


def infer_output_variable(_rules_inference):

    denominator = 0
    numerator = 0

    for i in range(len(_rules_inference)):
        numerator += _rules_inference[i][0] * output_var_centroids[_rules_inference[i][1]]
        denominator += _rules_inference[i][0]

    return numerator/denominator


# Calculating The Output Variable Centroids
for i in range(len(linguistics[-1])):
    output_var_centroids[linguistics[-1][i]] = centroid_x(linguistics_values[-1][i])

inferred_output_variable = infer_output_variable(rules_inference)

# print(variables)
# print(linguistics)
# print(linguistics_values)
# print(rules)
# print(operations)
print(variables_membership)
print(rules_inference)
print(output_var_centroids)
print(inferred_output_variable)



