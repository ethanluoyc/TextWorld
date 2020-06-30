
import json
import textwrap
from os.path import join as pjoin
from pprint import pprint

from collections import Counter, defaultdict, deque
from functools import total_ordering, lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Set, Sequence

from tatsu.model import NodeWalker

import fast_downward
from fast_downward import Operator

import textworld.logic.model
from textworld.utils import check_flag
from textworld.logic.model import GameLogicModelBuilderSemantics
from textworld.logic.parser import GameLogicParser
from textworld.logic import Proposition, Variable, Predicate, Rule, Placeholder

from textworld.textgen import TextGrammar


class _ModelConverter(NodeWalker):
    """
    Converts TatSu model objects to our types.
    """

    def __init__(self, logic=None):
        super().__init__()
        self._cache = {}
        self._logic = logic

    def _unescape(self, string):
        # Strip quotation marks
        return string[1:-1]

    def _unescape_block(self, string):
        # Strip triple quotation marks and dedent
        string = string[3:-3]
        return textwrap.dedent(string)

    def walk_list(self, l):
        return [self.walk(node) for node in l]

    def walk_ActionTypeNode(self, node):
        action = Action(
            name=node.name,
            template=self._unescape(node.template.template),
            feedback_rule=self._unescape(node.feedback.name),
            pddl=self._unescape_block(node.pddl.code) if node.pddl else "",
            grammar=self._unescape_block(node.grammar.code) if node.grammar else "{}"
        )

        return action

    def walk_Document2Node(self, node):
        actions = {}
        grammar = {}
        for part in node.parts:
            if isinstance(part, textworld.logic.model.ActionTypeNode):
                action = self.walk(part)
                actions[action.name.lower()] = action
                grammar_data = "\n".join(line for line in action.grammar.split("\n") if not line.lstrip().startswith("#"))
                grammar.update(json.loads(grammar_data))
            elif isinstance(part, textworld.logic.model.ActionGrammarNode):
                grammar.update(json.loads(self._unescape_block(part.code)))

        grammar = TextGrammar.parse(json.dumps(grammar))
        return actions, grammar


_PARSER = GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True)


def _parse_and_convert(*args, **kwargs):
    model = _PARSER.parse(*args, **kwargs)
    return _ModelConverter().walk(model)

def get_var_name(value):
    if value.lower() in ("i", "p"):
        return value.upper()

    return value


def get_var_type(value):
    if value.lower() in ("i", "p"):
        return value.upper()

    return value[0]


class Atom(fast_downward.Atom):
    """
    An Atom object contains the following fields:

    :param num: Operator ID
    :type num: int
    :param name: Operator name
    :type name: string

    ..warning this class must reflect the C struct in interface.cc.

    """
    @property
    def as_fact(self) -> Proposition:
        atom_type, rest = self.name.split(" ", 1)
        name, args = rest.split("(", 1)
        args = args[:-1].split(", ")
        arguments = [Variable(get_var_name(arg), get_var_type(arg)) for arg in args if arg]
        if atom_type == "NegatedAtom":
            name = "not_" + name

        return Proposition(name, arguments)

    def get_fact(self, name2type={}) -> Proposition:
        atom_type, rest = self.name.split(" ", 1)
        name, args = rest.split("(", 1)
        args = args[:-1].split(", ")
        arguments = [Variable(get_var_name(arg), name2type[arg]) for arg in args if arg]
        if atom_type == "NegatedAtom":
            name = "not_" + name

        return Proposition(name, arguments)


class Action:
    def __init__(self, name, template, pddl, grammar, feedback_rule):
       self.name = name
       self.feedback_rule = feedback_rule
       self.template = template
       self.pddl = pddl
       self.grammar = grammar


class GameLogic:
    """
    The logic for a game (types, rules, grammar, etc.).
    """

    def __init__(self):
        self.domain_filename = None
        self.domain = None
        self._document = ""
        self.grammar = TextGrammar()
        self.actions = {}
        self.types = textworld.logic.TypeHierarchy()

    def load_domain(self, filename):
        self.domain_filename = filename

        with open(filename) as f:
            self.domain = f.read()

    def import_twl2(self, filename):
        with open(filename) as f:
            document = f.read()

        actions, grammar = _parse_and_convert(document, rule_name="start2")
        self.actions.update(actions)
        self.grammar.update(grammar)


class State(textworld.logic.State):
    """
    The current state of a world.
    """

    def __init__(self, logic: GameLogic, facts: Iterable[Proposition] = None):
        self._logic = logic
        self._actions = {}

        self._facts = defaultdict(set)
        self._vars_by_name = {}
        self._vars_by_type = defaultdict(set)
        self._var_counts = Counter()

        self.downward_lib = fast_downward.load_lib()
        if facts:
            self.add_facts(facts)
            self._init_planner()

    def _init_planner(self, problem_filename=None):
        with open(problem_filename) as f:
            problem = f.read()

        self.task, self.sas = fast_downward.pddl2sas(self._logic.domain, problem, verbose=check_flag("TW_PDDL_DEBUG"))
        self._actions = {a.name: a for a in self.task.actions}

        # Import types from fastdownward
        for type_ in self.task.types:
            try:
                type_ = textworld.logic.Type(type_.name, type_.supertype_names)
                self._logic.types.add(type_)
            except ValueError:
                pass

        self.downward_lib.load_sas(self.sas.encode('utf-8'))

        self.name2type = {o.name: o.type_name for o in self.task.objects}
        def _atom2proposition(atom):
            if isinstance(atom, fast_downward.translate.pddl.conditions.Atom):
                if atom.predicate == "=":
                    return None

                return Proposition(atom.predicate, [Variable(arg, self.name2type[arg]) for arg in atom.args])

            elif isinstance(atom, fast_downward.translate.pddl.f_expression.Assign):
                if atom.fluent.symbol == "total-cost":
                    return None

                #name = "{}_{}".format(atom.fluent.symbol, atom.expression.value)
                name = "{}".format(atom.expression.value)

                if str.isdigit(name):  # Discard distance relations.
                    return None

                return Proposition(name, [Variable(arg, self.name2type[arg]) for arg in atom.fluent.args])


        facts = [_atom2proposition(atom) for atom in self.task.init]
        facts = list(filter(None, facts))
        self.add_facts(facts)

        state_size = self.downward_lib.get_state_size()
        atoms = (Atom * state_size)()
        self.downward_lib.get_state(atoms)
        for atom in atoms:
            fact = atom.get_fact(self.name2type)
            if fact.is_negation:
                self.add_fact(fact)


    @classmethod
    def from_pddl(cls, logic: GameLogic, problem_filename: str) -> "State":
        state = cls(logic, [])
        state._init_planner(problem_filename)

        state_size = state.downward_lib.get_state_size()
        atoms = (Atom * state_size)()
        state.downward_lib.get_state(atoms)
        facts = [atom.get_fact(state.name2type) for atom in atoms]
        facts = [fact for fact in facts if not fact.is_negation]
        state.add_facts(facts)
        return state

    def as_pddl(self):
        predicate = "({name} {params})"
        problem = textwrap.dedent("""\
        (define (problem textworld-game-1)
            (:domain textworld)
            (:objects {objects})
            (:init {init})
            (:goal
                (and {goal}))
        )
        """)

        def _format_proposition(fact):
            return predicate.format(
                name=fact.name,
                params=" ".join(fact.names)
            )

        problem_pddl = problem.format(
            objects=" ".join(sorted(set("{} - {}".format(arg.name, arg.type) for fact in self.facts for arg in fact.arguments))),
            init=textwrap.indent("\n" + "\n".join(_format_proposition(fact) for fact in self.facts), "        "),
            #goal=textwrap.indent("\n" + "\n".join(_format_proposition(fact) for fact in game.quests[0].win_events[0].condition.preconditions), "            "),
            goal="",
        )

        problem_pddl = problem_pddl.replace("'", "2")  # hack
        problem_pddl = problem_pddl.replace("/", "-")  # hack
        print(problem_pddl)
        return problem_pddl
        # with open("/tmp/textworld/problem.pddl", "w") as f:
        #     f.write(problem_pddl)

    def predicate2pddl(self, predicate: Predicate):
        pass

    def rule2pddl(self, rule: Rule):
        pass

    def domain_as_pddl(self):
        domain = textwrap.dedent("""\
        (define (domain textworld)
            (:requirements
            :typing)
            (:types {types})
            (:predicates {predicates})

        {actions}
        )
        """)
        predicate = "({name} {params})"
        action = textwrap.dedent("""\
        (:action {name}
            :parameters ({parameters})
            :precondition
                (and {preconditions})
            :effect
                (and {effects})
        )
        """)

        def _differentiate_type(types):
            seen = []
            for t in types:
                if t in seen:
                    t = t + "2"
                seen.append(t)
            return seen

        def _format_predicate(pred):
            if isinstance(pred, textworld.logic.Signature):
                return predicate.format(
                    name=pred.name,
                    params=" ".join("?{p} - {t}".format(p=p, t=t) for t, p in zip(pred.types, _differentiate_type(pred.types)))
                )

            return predicate.format(
                name=pred.name,
                params=" ".join("?{p}".format(p=n) for n in pred.names)
            )

        def _format_effects(rule):
            text = ""
            text += textwrap.indent("\n" + "\n".join(_format_predicate(p) for p in rule.added), "            ")
            text += textwrap.indent("\n" + "\n".join("(not {})".format(_format_predicate(p)) for p in rule.removed), "            ")
            return text

        predicates = []
        for pred in sorted(self._logic.predicates):
            predicates.append(_format_predicate(pred))

        actions = []
        for k in sorted(self._logic.rules):
            rule = self._logic.rules[k]
            actions.append(
                action.format(
                    name=rule.name,
                    parameters=" ".join("?{p} - {t}".format(p=p.name, t=p.type) for p in rule.placeholders),
                    preconditions=textwrap.indent("\n" + "\n".join(_format_predicate(p) for p in rule.preconditions), "            "),
                    effects=_format_effects(rule),
                )
            )

        domain_pddl = domain.format(
            types=textwrap.indent("\n" + "\n".join(self._logic.types._types), "        "),
            predicates=textwrap.indent("\n" + "\n".join(predicates), "        "),
            actions=textwrap.indent("\n".join(actions), "    "),
        )
        domain_pddl = domain_pddl.replace("'", "2")  # hack
        domain_pddl = domain_pddl.replace("/", "-")  # hack
        print(domain_pddl)
        return domain_pddl
        # with open("/tmp/textworld/domain.pddl", "w") as f:
        #     f.write(domain_pddl)

    def all_applicable_actions(self):
        # print("# Count operators")
        operator_count = self.downward_lib.get_applicable_operators_count()
        # print("# Count operators - done")

        operators = (Operator * operator_count)()
        # print("# Getting operators")
        self.downward_lib.get_applicable_operators(operators)
        # print("# Getting operators - done")
        self._operators = {op.id: op for op in operators}
        # pprint(self._operators)

        actions = []
        seen_operators = set()
        for operator in operators:
            if operator.name in seen_operators:
                continue

            seen_operators.add(operator.name)

            splits = operator.name.split()
            name, arguments = splits[0], splits[1:]

            action = textworld.logic.Action(name=name, preconditions=[], postconditions=[])
            action.id = operator.id
            action.mapping = {Placeholder(p.name.strip("?"), p.type_name): Variable(arg, p.type_name) for p, arg in zip(self._actions[name].parameters, arguments)}
            action.command_template = self._logic.actions[name].template#.format(**substitutions)
            action.feedback_rule = self._logic.actions[name].feedback_rule
            actions.append(action)

        return actions

    def apply(self, action):
        # TODO: convert textworld.logic.Action into operator id.
        op = self._operators[action.id]  # HACK: assume action is operator id for now.

        effects = (Atom * op.nb_effect_atoms)()
        self.downward_lib.apply_operator(op.id, effects)
        # pprint(list(str(atom.get_fact(self.name2type)) for atom in effects))

        # Update facts
        changes = []
        for effect in effects:
            prop = effect.get_fact(self.name2type)
            changes.append(prop)
            self.remove_fact(prop.negate())
            self.add_fact(prop)

        return changes

    def check_goal(self):
        return self.downward_lib.check_goal()

    def print_state(self):
        print("-= STATE =-")
        state_size = self.downward_lib.get_state_size()
        atoms = (Atom * state_size)()
        self.downward_lib.get_state(atoms)
        print("\n".join(sorted(str(atom.get_fact(self.name2type)) for atom in atoms)))

    # def all_assignments(self,
    #                     rule: Rule,
    #                     mapping: Mapping[Placeholder, Optional[Variable]] = None,
    #                     partial: bool = False,
    #                     allow_partial: Callable[[Placeholder], bool] = None,
    #                     ) -> Iterable[Mapping[Placeholder, Optional[Variable]]]:
    # """
    # Find all possible placeholder assignments that would allow a rule to be instantiated in this state.

    # Parameters
    # ----------
    # rule :
    #     The rule to instantiate.
    # mapping : optional
    #     An initial mapping to start from, constraining the possible instantiations.
    # partial : optional
    #     Whether incomplete mappings, that would require new variables or propositions, are allowed.
    # allow_partial : optional
    #     A callback function that returns whether a partial match may involve the given placeholder.

    # Returns
    # -------
    # The possible mappings for instantiating the rule.  Partial mappings requiring new variables will have None in
    # place of existing Variables.
    # """
    #     pass

    def _query(self, rule):

        with textworld.utils.make_temp_directory() as tmpdir:
            output_filename = pjoin(tmpdir, "output.sas")

            if problem_filename is None:
                problem_filename = pjoin(tmpdir, "problem.pddl")
                with open(problem_filename, "w") as f:
                    f.write(self.as_pddl())

            # TODO: make translate return the string instead of writing to a temp file.
            task = translate(self._logic.domain_filename, problem_filename, output_filename)
            self._actions = {a.name: a for a in task.actions}

            with open(output_filename) as f:
                sas = f.read()

            self.downward_lib.load_sas(sas.encode('utf-8'))

    def __del__(self):
        if hasattr(self, "downward_lib"):
            fast_downward.close_lib(self.downward_lib)
