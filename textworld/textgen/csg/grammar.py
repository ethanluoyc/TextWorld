import itertools
import json
from copy import deepcopy
from collections import defaultdict

from typing import Iterable, Optional, Tuple, List, Dict

from textworld.utils import check_flag


class Symbol:
    def __init__(self, symbol: str, context: Dict = {}):
        self.symbol = symbol
        self.context = context

    def __str__(self):
        return str(self.symbol)

    def copy(self, context):
        copy = deepcopy(self)
        copy.context = copy_context(context)
        return copy


class TerminalSymbol(Symbol):

    def __repr__(self):
        return "TerminalSymbol({!r})".format(self.symbol)

    def derive(self, context=None) -> List[Symbol]:
        # context = context or self.context
        return [self]


class NonterminalSymbol(Symbol):

    def derive(self, context=None) -> List[Symbol]:
        context = copy_context(context or self.context)  # TODO: no need for a copy here
        context["parent"] = self

        start = self.symbol
        rules = context["csg"]["production_rules"].get(str(start))
        if not rules:
            raise CSGUnknownSymbolError(start)

        applicable_rules = []
        for rule in rules:
            if rule.condition is None:  # TODO: assuming the unconditioned alternative is listed last.
                applicable_rules.append((rule, context))
                continue

            contexts = query(rule.condition, context)
            if len(contexts) == 1:
                applicable_rules.append((rule, contexts[0]))
                continue

            if len(contexts) > 1:
                # TODO: should we print a warning message about getting more than one context?
                applicable_rules.append((rule, context))
                # raise NotImplementedError("Tell Marc about this.")

        if len(applicable_rules) == 0:
            return []  # No applicable production rule could be found.

        rule, context = applicable_rules[0] # TODO: weighted sampling instead of picking the first alternative.
        symbols = deepcopy(rule.rhs)
        for symbol in symbols:
            symbol.context = copy_context(context)

        return symbols

    def __repr__(self):
        return "NonterminalSymbol('<{}>')".format(self.symbol)


def join(sep, iterable):
    iterable = iter(iterable)
    yield next(iterable)

    for e in iterable:
        yield sep
        yield e


def copy_context(context):
    return {
        "parent": context.get("parent"),  # TODO
        "csg": context.get("csg"),  # TODO
        "state": context["state"],
        "facts": context["facts"],
        "variables": deepcopy(context["variables"]),
        "mapping": deepcopy(context["mapping"]),
        "entity_infos": context["entity_infos"],
    }


def display_list(l, context):
    if len(l) == 0:
        return [NonterminalSymbol("list_empty", context)]

    if len(l) == 1:
        return [l[0]]

    list_separator = NonterminalSymbol("list_separator", context)
    list_last_separator = NonterminalSymbol("list_last_separator", context)
    # "#list_separator#".join(l[:-1]) + "#list_last_separator#" + l[-1]
    return list(join(list_separator, l[:-1])) + [list_last_separator] + [l[-1]]


def query(expression, context):
    from textworld.logic import Rule, dnf
    from textworld.textgen.csg import _parse_and_convert

    terms = _parse_and_convert(expression, rule_name="onlyExpression", trace=check_flag("TW_CSG_TRACE"))

    contexts = []
    for conjunction in dnf(terms):
        rule = Rule(
            name="query",
            preconditions=list(conjunction),
            postconditions=[],
        )

        # from textworld.logic import Predicate, Rule
        # rule = Rule(
        #     name="query",
        #     preconditions=[Predicate.parse(e.strip()) for e in expression.split("&")],
        #     postconditions=[],
        # )

        for mapping in context["state"].all_assignments(rule, context["mapping"]):
            context_ = copy_context(context)
            new_variables = {ph.name: context_["entity_infos"][var.name] for ph, var in mapping.items()}
            context_["variables"].update(new_variables)
            context_["mapping"].update(mapping)
            contexts.append(context_)

    return contexts


def evaluate(expression, context):
    #from textworld.logic import Predicate, Rule, _parse_and_convert, dnf
    from textworld.logic import Rule, dnf
    from textworld.textgen.csg import _parse_and_convert

    terms = _parse_and_convert(expression, rule_name="onlyExpression", trace=check_flag("TW_CSG_TRACE"))

    for conjunction in dnf(terms):
        rule = Rule(
            name="query",
            preconditions=list(conjunction),
            postconditions=[],
        )

        mappings = list(context["state"].all_assignments(rule, context["mapping"]))
        if len(mappings) > 1:
            print()
            raise NotImplementedError("Too many mappings", mappings)
            return True

        # if len(mappings) == 1:
        #     mapping = mappings[0]
        #     context_ =  copy_context(context)
        #     new_variables = {ph.name: context_["entity_infos"][var.name] for ph, var in mapping.items()}
        #     context_["variables"].update(new_variables)
        #     context_["mapping"].update(mapping)
        #     return context_

    return mappings


class EvalSymbol(Symbol):
    def __init__(self, expression: str, context: Dict = {}):
        super().__init__(expression, context)
        self.expression = expression

    def __repr__(self):
        return "EvalSymbol('{{{}}}')".format(str(self.expression))

    def derive(self, context=None):
        context = copy_context(context or self.context)  # TODO: no need for a copy here.
        context["parent"] = self
        locals().update(context["variables"])
        res = eval(self.expression)
        if isinstance(res, list):
            assert False
            return res

        return [TerminalSymbol(res)]


class ListSymbol:
    def __init__(self, symbols: List[Symbol], context: Dict = {}):
        self.symbols = symbols
        self.context = context

    def __str__(self):
        return "[" + ", ".join(map(str, self.symbols)) + "]"

    def __repr__(self):
        return "ListSymbol('{!r}')".format(self.symbols)

    def derive(self, context=None) -> List[Symbol]:
        context = copy_context(context or self.context)  # TODO: no need for a copy here.
        context["parent"] = self

        for symbol in self.symbols:
            symbol.context = copy_context(context)

        derivations = [symbol.derive() for symbol in self.symbols]
        list_derivation = list(itertools.chain(*[derivation if isinstance(derivation, list) else [derivation] for derivation in derivations]))
        # list_derivation = [derivation if isinstance(derivation, list) else [derivation] for derivation in derivations if derivation]
        return display_list(list_derivation, context)

    def copy(self, context):
        copy = deepcopy(self)
        copy.context = copy_context(context)
        return copy


class ConditionalSymbol(Symbol):

    def __init__(self, expression: Symbol, given: str, context: Dict = {}):
        super().__init__(str(expression), context)
        self.expression = expression
        self.given = given

    def __repr__(self):
        return "ConditionalSymbol('{{{}|{}}}')".format(str(self.expression), str(self.given))

    def derive(self, context=None) -> List[Symbol]:
        context = copy_context(context or self.context)
        context["parent"] = self

        if len(self.context) == 0:
            raise ValueError("Empty context")

        contexts = [context]
        if self.given:
            contexts = query(self.given, context)

        res = [self.expression.copy(context) for context in contexts]
        return res


class ProductionRule:
    """ Production rule for a context-sensitive grammar. """

    # TODO: support multiple symbols for the rhs?
    def __init__(self, lhs: str, rhs: List[Symbol], weight=1, condition=None):
        """
        Arguments:
            rhs: symbol that will be transformed by this production rule.
            lhs: list of symbols generated by this production rule.
            weight: prevalence of this production.
        """
        self.lhs = lhs
        self.rhs = rhs
        self.weight = weight
        self.condition = condition

    def __repr__(self):
        return "ProductionRule(lhs={!r}, rhs={!r}, weight={!r}, condition={!r})".format(self.lhs, self.rhs, self.weight, self.condition)


class CSGUnknownSymbolError(Exception):
    def __init__(self, symbol: Symbol):
        msg = "Can't find symbol '<{}>' in the set of production rules."
        # TODO: mention closest match?
        super().__init__(msg.format(symbol))


class ContextSensitiveGrammar:

    def __init__(self):
        self._rules = defaultdict(list)
        self.actions = {}

    def update(self, grammar):
        for k, v in grammar._rules.items():
            self._rules[k].extend(v)

    @classmethod
    def load(cls, document: str):
        from textworld.textgen.csg import _PARSER, _Converter
        grammar = cls()
        model = _PARSER.parse(document, colorize=check_flag("TW_CSG_TRACE"), trace=check_flag("TW_CSG_TRACE"))
        return _Converter(grammar).walk(model)

    # @classmethod
    # def parse(cls, text: str):#, filename: Optional[str] = None):
    #     data = json.loads(text)

    #     grammar = cls()
    #     for name, rules in data.items():
    #         rules = [ProductionRule(lhs=name,
    #                                 rhs=_parse_and_convert(rule["rhs"], rule_name="String"),
    #                                 weight=rule.get("weight", 1),
    #                                 condition=rule.get("condition", ""))
    #                  for rule in rules]

    #         for rule in rules:
    #             # print(repr(rule))
    #             grammar.add_rule(rule)

    #     # model = _PARSER.parse(text, filename=filename)
    #     # _Converter(grammar).walk(model)
    #     return grammar

    def add_rule(self, rule: ProductionRule):
        self._rules[rule.lhs].append(rule)

    def add_rules(self, rules: List[ProductionRule]):
        for rule in rules:
            self.add_rule(rule)

    # def replace(self, start: Symbol) -> List[Symbol]:
    #     rules = self._rules.get(str(start))
    #     if not rules:
    #         raise CSGUnknownSymbolError(start)

    #     def _applicable(rule):
    #         if not rule.condition:
    #             return True

    #         return evaluate(rule.condition, start.context)

    #     rules = list(filter(_applicable, rules))

    #     # TODO: deal with multiple alternatives
    #     # TODO: deal with context
    #     symbols = deepcopy(rules[0].rhs)
    #     for symbol in symbols:
    #         symbol.context = copy_context(start.context)

    #     return symbols

    def derive(self, start: str, context={}) -> str:
        from textworld.textgen.csg import _parse_and_convert
        derivation = _parse_and_convert(start, rule_name="String", trace=check_flag("TW_CSG_TRACE"))
        derivation = derivation[::-1]  # Reverse to build a derivation stack.

        context["csg"] = {"production_rules": self._rules}
        context["parent"] = start
        for symbol in derivation:
            symbol.context = copy_context(context)

        derived = []
        while len(derivation) > 0:
            if check_flag("TW_CSG_DEBUG"):
                print(derivation)

            symbol = derivation.pop()
            if isinstance(symbol, TerminalSymbol):
                derived.append(symbol)

            # elif isinstance(symbol, NonterminalSymbol):
            #     derivation += self.replace(symbol)[::-1]  # Reverse to add on top of the derivation stack.

            # elif isinstance(symbol, (ConditionalSymbol, EvalSymbol, ListSymbol)):
            elif isinstance(symbol, (NonterminalSymbol, ConditionalSymbol, EvalSymbol, ListSymbol)):
                derivation += list(itertools.chain(*[s if isinstance(s, list) else [s] for s in symbol.derive()]))[::-1]  # Reverse to add on top of the derivation stack.

            else:
                raise NotImplementedError("Unknown symbol: {}".format(type(symbol)))

        return "".join(map(str, derived))
