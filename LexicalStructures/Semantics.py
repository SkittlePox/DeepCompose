import enum


class SemanticTypePrimitive(enum.Enum):
    def __str__(self):
        return str(self.value)

    e = "e"  # Individual
    t = "t"  # Truth value


class SemanticType:
    """
    This object contains a semantic type like <e,t>
    A linguistic expression with type <e,t> is a function from individuals (e) to truth values (t)
    """

    def __init__(self, lhs=None, rhs=None):
        self.lhs = lhs
        self.rhs = rhs

    def complexity(self):
        """
        This is for sorting purposes
        """
        if self.lhs == None:
            return 1
        lhs = 1 if isinstance(self.lhs, SemanticTypePrimitive) else self.lhs.complexity()
        rhs = 1 if isinstance(self.rhs, SemanticTypePrimitive) else self.rhs.complexity()
        return lhs + rhs

    def __call__(self, argument):
        return self.rhs

    def __str__(self):
        if self.lhs == None:
            return str(self.rhs)
        else:
            return f"<{str(self.lhs)},{str(self.rhs)}>"

    def __eq__(self, other):
        return isinstance(other, SemanticType) \
               and self.lhs == other.lhs and self.rhs == other.rhs


#########################

class LambdaCalcExpression:
    def __init__(self, expression):
        """
        Expression may be a callable function or a dictionary to be turned into a callable function
        This class represents a lambda calculus expression, intention
        """

        def dict_to_func(dictionary):
            def get_arg(argument):
                if argument in dictionary:
                    return LambdaCalcExpression(expression=dictionary[argument])
                else:
                    return LambdaCalcExpression(expression={})

            return get_arg

        self.expression = expression
        if isinstance(self.expression, dict):
            self.function = dict_to_func(self.expression)
        else:
            self.function = self.expression

    def update(self, new_entries):
        def dict_updater(orig_dict, entries):
            for key, val in entries.items():
                if isinstance(val, dict):
                    dict_updater(orig_dict[key], val)
                else:
                    orig_dict.update({key: val})

        dict_updater(self.expression, new_entries)

    def __str__(self):
        def dict_to_special(dictionary):
            baseStr = ""
            for key, value in dictionary.items():
                if isinstance(value, str):
                    baseStr += f"{key}={str(value)} "
                else:
                    baseStr += f"{key}=({dict_to_special(value)}) "
            baseStr = baseStr[:-1]
            return baseStr

        def func_to_special(function):
            baseStr = ""
            for arg in self.arguments:
                baseStr += f"{arg}=({str(function(arg))}) "
            baseStr = baseStr[:-1]
            return baseStr

        if isinstance(self.expression, str) or isinstance(self.expression, int):
            return str(self.expression)
        elif not self.expression:
            return "undefined"
        elif isinstance(self.expression, dict):
            return dict_to_special(self.expression)
        else:
            return "special function"

    def __call__(self, argument):
        return self.function(argument.expression)


#########################

class SemanticIntension:
    def __init__(self, function=None, argument=None):
        """
        Takes a function and argument

        function: This might look like 'likes(a)'
        argument: This might look like 'b'

        Together this intension would represent 'likes(a)(b)'

        In the case of a primitive like 'charlie'
        function = None (This should probably be some identity function)
        argument = 'charlie'
        """
        self.function = function
        self.argument = argument

    def __call__(self, argument):
        return SemanticIntension(function=self, argument=argument)

    def __str__(self):
        if self.function is None:
            return str(self.argument)
        else:
            return f"{str(self.function)}({str(self.argument)})"


#########################

class SemanticEntry:
    def __init__(self, id, intension=None, extension=None, semantic_type=None):
        """
        Takes a SemanticIntention, SemanticExtension, and SemanticType
        """
        self.id = id
        self.intension = intension
        self.extension = extension
        self.semantic_type = semantic_type

    def update(self, new_entries):
        """
        This allows for the possibility of re-defining expressions
        """
        self.extension.update(new_entries)

    def complexity(self):
        return self.semantic_type.complexity()

    def __call__(self, argument):
        """
        argument is another SemanticEntry
        """
        intension = None if self.intension is None else self.intension(argument.intension)
        extension = None if self.extension is None else self.extension(argument.extension)
        semantic_type = self.semantic_type(argument.semantic_type)
        return SemanticEntry(id=f"{self.id}({argument.id})", intension=intension, extension=extension, semantic_type=semantic_type)

    def __str__(self):
        baseStr = ""
        baseStr += str(self.semantic_type)
        baseStr += " ; "
        baseStr += str(self.id)
        # baseStr += " ; "
        # baseStr += str(self.intension)
        # baseStr += " ; "
        # baseStr += str(self.extension)
        return baseStr
