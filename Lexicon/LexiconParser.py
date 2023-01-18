from .Syntax import *
from .Semantics import *
from .LexicalStructures import *


class LexiconParser:
    def __init__(self):
        pass

    def parse_syntactic_category(self, cat):
        """
        Parses a syntactic category like S/NP or (S/NP)/(S/NP)
        """

        def find_slash(subcat):
            bracket_count = 0
            for i in range(len(subcat)):
                if subcat[i] == "(":
                    bracket_count += 1
                elif subcat[i] == ")":
                    bracket_count -= 1
                elif bracket_count == 0 and subcat[i] == "/":
                    return i

        def strip_feature(subcat):
            features = []
            category = ""
            have_cat = False
            feature_start_ind = 0
            for i in range(len(subcat)):
                if subcat[i] == "[":
                    if not have_cat:
                        have_cat = True
                        category = subcat[:i]
                    feature_start_ind = i + 1
                elif subcat[i] == "]":
                    features.append(subcat[feature_start_ind:i])
            return category, features

        if "/" not in cat:
            if "[" not in cat:
                return SyntacticCategory(lhs=SyntacticPrimitive(cat))
            else:
                cat, features = strip_feature(cat)
                return SyntacticCategory(lhs=SyntacticPrimitive(cat), features=features)
        else:
            center_slash = find_slash(cat)
            first_arg = cat[:center_slash]
            slash = None

            if cat[center_slash + 1] == "r":
                second_arg = cat[center_slash + 2:]
                slash = SyntacticSlash.R
            elif cat[center_slash + 1] == "l":
                second_arg = cat[center_slash + 2:]
                slash = SyntacticSlash.L
            else:
                second_arg = cat[center_slash + 1:]

            if first_arg[0] == "(":
                first_arg = first_arg[1:-1]
            if second_arg[0] == "(":
                second_arg = second_arg[1:-1]
            return SyntacticCategory(lhs=self.parse_syntactic_category(first_arg),
                                     rhs=self.parse_syntactic_category(second_arg), slash=slash)

    def parse_semantic_type(self, semantic_type):
        """
        Parses a semantic type like <e,t> or <e,<e,t>>
        """

        def find_break(s_type):
            bracket_count = 0
            for i in range(len(s_type)):
                if s_type[i] == "<":
                    bracket_count += 1
                elif s_type[i] == ">":
                    bracket_count -= 1
                elif bracket_count == 0 and s_type[i] == ",":
                    return i

        if semantic_type[0] != "<":
            return SemanticType(rhs=SemanticTypePrimitive(semantic_type))
        else:
            subtype = semantic_type[1:-1]
            centerline = find_break(subtype)
            return SemanticType(lhs=self.parse_semantic_type(subtype[:centerline]),
                                rhs=self.parse_semantic_type(subtype[centerline + 1:]))

    def parse_semantic_extension(self, func):
        """
        Parses a semantic function into a dictionary
        m=(m=1 p=1) p=(m=0 p=1) z=(m=0 p=0) -> {m: {m: 1, p: 1}, p: {m: 0, p: 1}, z: {m: 0, p: 0}}
        """

        def tokenize(s_func):  # This tokenizes a function: m=1 p=0 -> ["m=1", "p=0"]
            s_func += " "
            bracket_count = 0
            start = 0
            lst = []
            i = 0
            while i < len(s_func):
                if s_func[i] == "(":
                    bracket_count += 1
                elif s_func[i] == ")":
                    bracket_count -= 1
                elif bracket_count == 0 and s_func[i] == " " or i == len(s_func) - 1:
                    lst.append(s_func[start:i])
                    start = i + 1
                    i += 1
                i += 1
            return lst

        def to_dict_entry(entry):
            if "(" not in entry:
                equals = entry.index('=')
                return {entry[:equals]: entry[equals + 1:]}
            else:
                equals = entry.index('=')
                return {entry[:equals]: self.parse_semantic_extension(entry[equals + 2:])}

        tokens = tokenize(func)
        if len(tokens) == 1 and "=" not in tokens[0]:
            return tokens[0]
        else:
            func_dict = {}
            entries = list(map(to_dict_entry, tokens))
            for e in entries:
                func_dict.update(e)
            return func_dict

    def parse_entry(self, entry):
        """
        Parses an individual lexical entry in a lexicon file
        The parsing is broken up into syntactic, semantic, etc.
        """
        entry_array = entry.split(" ; ")
        english = entry_array[0]
        category = self.parse_syntactic_category(entry_array[1])
        semantic_type = self.parse_semantic_type(entry_array[2])
        semantics = SemanticIntensionPrimitive(name=entry_array[3], module=ExtensionModule(
            output_dims=get_semantic_type_dims(semantic_type)),
                                               semantic_type=semantic_type)
        return LexicalEntry(english, category, semantics)

    def parse_file(self, filename):
        """
        Turns a lexicon file into a list of LexicalEntry objects
        """
        entries = []
        with open(filename, 'r') as fp:
            line = fp.readline()
            while line:
                if line[0] != "#":
                    entries.append(self.parse_entry(line[:-1]))
                line = fp.readline()
        return entries
