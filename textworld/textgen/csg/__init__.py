import itertools
import json
from copy import deepcopy
from collections import defaultdict

from tatsu.model import NodeWalker
from typing import Iterable, Optional, Tuple, List, Dict

from textworld.utils import check_flag

from textworld.logic import And, Or, Predicate, Placeholder

from textworld.textgen.csg.model import TWL2ModelBuilderSemantics
from textworld.textgen.csg.parser import TWL2Parser

from textworld.textgen.csg.grammar import ContextSensitiveGrammar
from textworld.textgen.csg.grammar import TerminalSymbol, NonterminalSymbol, ConditionalSymbol
from textworld.textgen.csg.grammar import EvalSymbol, ListSymbol
from textworld.textgen.csg.grammar import ProductionRule



class _Converter(NodeWalker):

    def __init__(self, grammar: Optional["ContextSensitiveGrammar"] = None):
        self.grammar = grammar

    def walk_list(self, node):
        return [self.walk(child) for child in node]

    def walk_str(self, node):
        return node.replace("\\n", "\n")

    def walk_TerminalSymbol(self, node):
        return TerminalSymbol(self.walk(node.literal))

    def walk_NonterminalSymbol(self, node):
        return NonterminalSymbol(node.symbol)

    def walk_ConditionalSymbol(self, node):
        return ConditionalSymbol(self.walk(node.expression), node.given)

    def walk_SpecialSymbol(self, node):
        return self.walk(node.statement)

    def walk_EvalSymbol(self, node):
        return EvalSymbol(node.statement)

    def walk_ListSymbol(self, node):
        return ListSymbol(self.walk(node.symbols))

    def walk_String(self, node):
        return self.walk(node.symbols)

    def walk_PlaceholderNode(self, node):
        return Placeholder(node.name, type="object")

    def walk_PredicateNode(self, node):
        pred = Predicate(node.name.lstrip("!"), self.walk(node.parameters))
        if node.name.startswith("!"):
            return pred.negate()

        return pred

    def walk_ExpressionNode(self, node):
        return self.walk(node.expression)

    def walk_ConjunctionNode(self, node):
        return And(self.walk(node.expressions))

    def walk_DisjunctionNode(self, node):
        return Or(self.walk(node.expressions))

    def walk_ProductionRule(self, node):
        for string in node.alternatives:
            if string is None:
                assert False
                continue

            rule = ProductionRule(node.symbol, self.walk(string))
            self.grammar.add_rule(rule)

    def walk_ActionNode(self, node):
        node.name = self.walk(node.name.name)
        node.template = self.walk(node.template)
        node.feedback = self.walk(node.feedback)
        return node

    def walk_ActionsNode(self, node):
        actions = {action.name: action for action in self.walk(node.actions)}
        self.grammar.actions.update(actions)
        return actions

    def walk_RhsNode(self, node):
        return self.walk(node.symbols), node.given, 1  # TODO: add support for weighted alternatives.

    def walk_RuleNode(self, node):
        lhs = node.lhs.name
        if not isinstance(node.rhs, list):
            node.rhs = [node.rhs]

        rules = []
        for rhs, condition, weight in self.walk(node.rhs):
            rules.append(ProductionRule(lhs, rhs, weight, condition))

        return rules

    def walk_GrammarNode(self, node):
        rules = self.walk(node.rules)
        self.grammar.add_rules(itertools.chain(*rules))
        return self.grammar

    def walk_TWL2Document(self, node):
        self.walk(node.grammar)
        self.walk(node.actions)
        return self.grammar


_PARSER = TWL2Parser(semantics=TWL2ModelBuilderSemantics(), parseinfo=False)


def _parse_and_convert(*args, **kwargs):
    model = _PARSER.parse(*args, **kwargs)
    return _Converter().walk(model)
