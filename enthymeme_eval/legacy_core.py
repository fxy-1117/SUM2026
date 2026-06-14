# This module contains the legacy symbolic/AMR logic extracted from Enthymem-newtry-two.ipynb.
# Clean wrappers live in the neighboring modules; keep this file close to notebook behavior.


# ===== notebook cell 0 =====
import re
from operator import itemgetter
from amr_logic_converter import types
from sympy.logic.boolalg import to_cnf
from sympy import Symbol
from pysat.formula import CNF
from pysat.solvers import Solver
# import spacy
# nlp = spacy.load("en_core_web_sm")

# ===== notebook cell 3 =====
from amr_logic_converter.types import *
from typing import Optional, Dict
def strip_suffix(symbol: str) -> str:
    """
    Removes numerical suffixes and slashes from a predicate symbol.
    Examples:
        'cause-01' -> 'cause'
        'spread/03' -> 'spread'
    """
    # Remove any '-' or '/' followed by digits at the end of the string
    return re.sub(r'[-/]\d+$', '', symbol)
def transform_replace_constants(clause: Clause) -> Optional[Clause]:
    """
    Transforms the Clause by replacing constants with their corresponding predicates
    based on type-defining atoms and removes type-defining atoms from the Clause.

    Example:
        man(m) ∧ mod(g, m) ∧ good(g) → mod(good, man)
    """
    # Step 1: Collect mappings from terms to predicates based on type-defining atoms
    term_to_predicate: Dict[str, str] = {}
    type_defining_predicates: set[str] = set()

    def collect_mappings(current_clause: Clause):
        nonlocal term_to_predicate, type_defining_predicates
        if isinstance(current_clause, Atom):
            if len(current_clause.terms) == 1:
                term = current_clause.terms[0]
                if isinstance(term, Constant):
                    term_to_predicate[term.value] = strip_suffix(current_clause.predicate.symbol)
                    type_defining_predicates.add(current_clause.predicate.symbol)
        elif isinstance(current_clause, And) or isinstance(current_clause, Or):
            for arg in current_clause.args:
                collect_mappings(arg)
        elif isinstance(current_clause, Not):
            collect_mappings(current_clause.body)
        elif isinstance(current_clause, Implies):
            collect_mappings(current_clause.antecedent)
            collect_mappings(current_clause.consequent)
        elif isinstance(current_clause, Exists) or isinstance(current_clause, All):
            collect_mappings(current_clause.body)
        # Add more Clause types if necessary

    collect_mappings(clause)

    # Step 2: Replace terms in Atoms based on the mapping and remove type-defining Atoms
    def replace_terms(current_clause: Clause) -> Optional[Clause]:
        if isinstance(current_clause, Atom):
            # Skip type-defining atoms
            if current_clause.predicate.symbol in type_defining_predicates and len(current_clause.terms) == 1:
                return None  # Remove this Atom

            # Replace terms if they are in the mapping
            new_terms = []
            for term in current_clause.terms:
                if isinstance(term, Constant) and term.value in term_to_predicate:
                    # Replace with corresponding predicate
                    new_pred_symbol = term_to_predicate[term.value]
                    new_predicate = Predicate(symbol=new_pred_symbol)
                    new_terms.append(new_predicate)
                elif isinstance(term, Variable) and term.name in term_to_predicate:
                    # Replace with corresponding predicate if Variables can be mapped
                    new_pred_symbol = term_to_predicate[term.name]
                    new_predicate = Predicate(symbol=new_pred_symbol)
                    new_terms.append(new_predicate)
                else:
                    # Keep the term as is
                    new_terms.append(term)

            # After replacement, ensure dyadic predicates
            if len(new_terms) <= 2:
                return Atom(predicate=current_clause.predicate, terms=tuple(new_terms))
            else:
                # Break down into dyadic atoms connected by And
                dyadic_atoms = []
                first_term = new_terms[0]
                for term in new_terms[1:]:
                    dyadic_atoms.append(Atom(predicate=current_clause.predicate, terms=(first_term, term)))
                return And(*dyadic_atoms)

        elif isinstance(current_clause, And):
            # Recursively transform each argument
            transformed_args = []
            for arg in current_clause.args:
                transformed = replace_terms(arg)
                if transformed is not None:
                    transformed_args.append(transformed)
            if not transformed_args:
                return None
            elif len(transformed_args) == 1:
                return transformed_args[0]
            else:
                return And(*transformed_args)

        elif isinstance(current_clause, Or):
            # Recursively transform each argument
            transformed_args = []
            for arg in current_clause.args:
                transformed = replace_terms(arg)
                if transformed is not None:
                    transformed_args.append(transformed)
            if not transformed_args:
                return None
            elif len(transformed_args) == 1:
                return transformed_args[0]
            else:
                return Or(*transformed_args)

        elif isinstance(current_clause, Not):
            # Recursively transform the body
            transformed_body = replace_terms(current_clause.body)
            if transformed_body is None:
                return None
            return Not(body=transformed_body)

        elif isinstance(current_clause, Implies):
            # Recursively transform antecedent and consequent
            transformed_antecedent = replace_terms(current_clause.antecedent)
            transformed_consequent = replace_terms(current_clause.consequent)
            if transformed_antecedent is None or transformed_consequent is None:
                return None
            return Implies(antecedent=transformed_antecedent, consequent=transformed_consequent)

        elif isinstance(current_clause, Exists):
            # Recursively transform the body
            transformed_body = replace_terms(current_clause.body)
            if transformed_body is None:
                return None
            return Exists(param=current_clause.param, body=transformed_body)

        elif isinstance(current_clause, All):
            # Recursively transform the body
            transformed_body = replace_terms(current_clause.body)
            if transformed_body is None:
                return None
            return All(param=current_clause.param, body=transformed_body)

        else:
            raise TypeError(f"Unsupported Clause type: {type(current_clause)}")

    transformed_clause = replace_terms(clause)
    return transformed_clause


def merge_quant_predicates(clause: Clause) -> Clause:
    """
    Recursively merges :mod predicates into their corresponding predicates within any And clause.
    Supports multiple :mod predicates and multiple corresponding predicates, even within nested clauses.

    Args:
        clause (Clause): The clause to process.

    Returns:
        Clause: The transformed clause with merged predicates.
    """
    if isinstance(clause, And):
        # Step 1: Identify all :mod predicates within this And clause
        mod_atoms = [
            atom for atom in clause.args
            if isinstance(atom, Atom) and atom.predicate.symbol == ":quant"
        ]

        # Step 2: Create a mapping from X to Y for :mod(X, Y)
        mod_mapping: Dict[str, str] = {}
        for atom in mod_atoms:
            if len(atom.terms) != 2:
                continue  # Ignore malformed :mod predicates
            X, Y = atom.terms
            X_str = str(X)
            Y_str = str(Y)
            mod_mapping[X_str] = Y_str

        # Step 3: Process clauses in their original order
        new_args: List[Clause] = []
        for arg in clause.args:
            if isinstance(arg, Atom):
                if arg.predicate.symbol == ":quant":
                    # Skip :mod atoms as they've been processed
                    continue
                else:
                    # Apply mapping to terms if applicable
                    new_terms = []
                    for term in arg.terms:
                        term_str = str(term)
                        if term_str in mod_mapping:
                            # Merge Y and X to form "Y X" without quotes
                            merged_term_str = f"{mod_mapping[term_str]} {term_str}"
                            # Create a new Constant with type="symbol" to avoid quotes
                            merged_term = Constant(element=merged_term_str, type="symbol")
                            new_terms.append(merged_term)
                        else:
                            new_terms.append(term)
                    # Create a new Atom with updated terms
                    new_atom = Atom(arg.predicate, tuple(new_terms))
                    # Recursively process in case there are nested :mod predicates
                    merged_atom = merge_quant_predicates(new_atom)
                    new_args.append(merged_atom)
            else:
                # For non-Atom clauses (e.g., Not, Or, etc.), recursively process
                processed_clause = merge_quant_predicates(arg)
                new_args.append(processed_clause)

        # Step 4: Reconstruct the And clause without :mod predicates
        return And(*new_args)

    elif isinstance(clause, Or):
        # Recursively process each argument
        processed_args = [merge_quant_predicates(arg) for arg in clause.args]
        return Or(*processed_args)

    elif isinstance(clause, Not):
        # Recursively process the body of the Not
        processed_body = merge_quant_predicates(clause.body)
        return Not(processed_body)

    elif isinstance(clause, Implies):
        # Recursively process antecedent and consequent
        processed_antecedent = merge_quant_predicates(clause.antecedent)
        processed_consequent = merge_quant_predicates(clause.consequent)
        return Implies(processed_antecedent, processed_consequent)

    elif isinstance(clause, Exists):
        # Recursively process the body of Exists
        processed_body = merge_quant_predicates(clause.body)
        return Exists(clause.param, processed_body)

    elif isinstance(clause, All):
        # Recursively process the body of All
        processed_body = merge_quant_predicates(clause.body)
        return All(clause.param, processed_body)

    elif isinstance(clause, Atom):
        # Base case: Atom without any further processing needed
        return clause

    else:
        # For unsupported types, return as is
        return clause
def merge_mod_predicates(clause: Clause) -> Clause:
    """
    Recursively merges :mod predicates into their corresponding predicates within any And clause.
    Maintains separate instances for different modifiers of the same term.
    """
    if isinstance(clause, And):
        # Step 1: Identify all :mod predicates within this And clause
        mod_atoms = [
            atom for atom in clause.args
            if isinstance(atom, Atom) and atom.predicate.symbol == ":mod"
        ]

        # Step 2: Create a mapping from X to ordered list of modifiers for :mod(X, Y)
        mod_mapping: Dict[str, List[str]] = {}
        for atom in mod_atoms:
            if len(atom.terms) != 2:
                continue
            X, Y = atom.terms
            X_str = str(X)
            Y_str = str(Y)
            if X_str not in mod_mapping:
                mod_mapping[X_str] = []
            mod_mapping[X_str].append(Y_str)

        # Step 3: Process clauses in their original order
        new_args: List[Clause] = []
        
        for arg in clause.args:
            if isinstance(arg, Atom):
                if arg.predicate.symbol == ":mod":
                    continue
                
                new_terms = []
                for term in arg.terms:
                    term_str = str(term)
                    if term_str in mod_mapping:
                        # Create merged term with modifiers in reverse order
                        # Last modifier should be closest to the term
                        modifiers = list(reversed(mod_mapping[term_str]))
                        base_term = term_str
                        for modifier in modifiers:
                            base_term = f"{modifier} {base_term}"
                        merged_term = Constant(element=base_term, type="symbol")
                        new_terms.append(merged_term)
                    else:
                        new_terms.append(term)
                
                new_atom = Atom(arg.predicate, tuple(new_terms))
                merged_atom = merge_mod_predicates(new_atom)
                new_args.append(merged_atom)
            else:
                processed_clause = merge_mod_predicates(arg)
                new_args.append(processed_clause)

        return And(*new_args)

    # Rest of the function remains the same
    elif isinstance(clause, Or):
        processed_args = [merge_mod_predicates(arg) for arg in clause.args]
        return Or(*processed_args)
    elif isinstance(clause, Not):
        processed_body = merge_mod_predicates(clause.body)
        return Not(processed_body)
    elif isinstance(clause, Implies):
        processed_antecedent = merge_mod_predicates(clause.antecedent)
        processed_consequent = merge_mod_predicates(clause.consequent)
        return Implies(processed_antecedent, processed_consequent)
    elif isinstance(clause, Exists):
        processed_body = merge_mod_predicates(clause.body)
        return Exists(clause.param, processed_body)
    elif isinstance(clause, All):
        processed_body = merge_mod_predicates(clause.body)
        return All(clause.param, processed_body)
    elif isinstance(clause, Atom):
        return clause
    else:
        return clause
def remove_duplicate_predicates(clause: Clause) -> Clause:
    """
    Removes duplicate predicates within And clauses while preserving order of first appearance.
    """
    if isinstance(clause, And):
        # Use a dictionary to track unique predicates
        # Key: (predicate_symbol, terms_tuple)
        # Value: Atom
        unique_predicates: Dict[str, Atom] = {}
        
        # Process all clauses in order
        for arg in clause.args:
            if isinstance(arg, Atom):
                # Create a unique key for this predicate
                # Convert terms to strings and join them to create a unique identifier
                key = f"{arg.predicate.symbol}|{','.join(str(term) for term in arg.terms)}"
                
                # Only keep the first occurrence
                if key not in unique_predicates:
                    unique_predicates[key] = arg
            else:
                # For non-Atom clauses, process recursively
                processed_clause = remove_duplicate_predicates(arg)
                # Create a special key for non-Atom clauses
                key = f"non_atom|{str(processed_clause)}"
                if key not in unique_predicates:
                    unique_predicates[key] = processed_clause

        # Reconstruct the And clause with unique predicates in original order
        new_args = []
        seen_keys = set()
        
        # Preserve original order by iterating through original args
        for arg in clause.args:
            if isinstance(arg, Atom):
                key = f"{arg.predicate.symbol}|{','.join(str(term) for term in arg.terms)}"
            else:
                processed_arg = remove_duplicate_predicates(arg)
                key = f"non_atom|{str(processed_arg)}"
            
            if key not in seen_keys:
                new_args.append(unique_predicates[key])
                seen_keys.add(key)

        return And(*new_args)

    # Handle other clause types recursively
    elif isinstance(clause, Or):
        processed_args = [remove_duplicate_predicates(arg) for arg in clause.args]
        return Or(*processed_args)
    elif isinstance(clause, Not):
        processed_body = remove_duplicate_predicates(clause.body)
        return Not(processed_body)
    elif isinstance(clause, Implies):
        processed_antecedent = remove_duplicate_predicates(clause.antecedent)
        processed_consequent = remove_duplicate_predicates(clause.consequent)
        return Implies(processed_antecedent, processed_consequent)
    elif isinstance(clause, Exists):
        processed_body = remove_duplicate_predicates(clause.body)
        return Exists(clause.param, processed_body)
    elif isinstance(clause, All):
        processed_body = remove_duplicate_predicates(clause.body)
        return All(clause.param, processed_body)
    elif isinstance(clause, Atom):
        return clause
    else:
        return clause



# ===== notebook cell 4 =====
def generate_logic(data):
    tem  = []
    temm = []
    tem_token = []
    for sen in data:
        tokens, _ = parser.tokenize(sen)
        tem_token.append(tokens)
    
    annotations, machines = parser.parse_sentences(tem_token)
    tem = annotations
    temm = [i.get_amr().to_penman(jamr=False, isi=False) for i in machines]
    n = 0
    r1 = []
    r2 = []
    for sen in data:
        r1.append(converter.convert(tem[n]))
        r2.append(converter.convert(temm[n]))
        n+=1
    return tem,temm, r1,r2

# ===== notebook cell 5 =====
def combine(final,f = False):
    init = True
    for i in final:
        if type(i) == list:
            tem = True
            
            tem = tem&combine(i)
            if ~tem == -1:
                
                init = init & True
            elif ~tem == -2:
                if not f:
                    init = init & False
                else:
                    init = init & True
            else:
                init = init&~tem
        else:
            init = init&i
            
    return init
    

# ===== notebook cell 6 =====
import copy
def transform(formula,X):
    final = copy.deepcopy(formula)
    for i in range(len(final)):
        
        if type(final[i]) == list:
            
            if final[i][0] == "ARG":
                if "/".join([final[i][1],final[i][2],final[i][3]]) not in X:
                    continue
                else:
                    final[i] = X["/".join([final[i][1],final[i][2],final[i][3]])]
#             
                
#                     init = init
            else:
                final[i] = transform(final[i],X)
               
#                     else:
                    
        else:
            if final[i] not in X:
                continue
        
            else:
                final[i]  = X[final[i]] 

    
    return final
        

# ===== notebook cell 7 =====
def extract(formula,l= 0):
    and_list = []
    arg = []
    if type(formula) == types.Not:
#         return "neg","neg","neg"
        return [extract(formula.body)[0]],arg+extract(formula.body,1)[1]
    if type(formula) != types.Atom:
        for i in formula.args:
            if type(i) == types.Not:
#             return "neg","neg","neg"
                and_list.append(extract(i.body)[0])
                arg = arg+extract(i.body,1)[1]
           
            else:
                tem = []
                for j in range(0,len(i.terms)):
                    try:
                        tem.append(i.terms[j].symbol)
                    except:
                        tem.append(i.terms[j].value)
            
                and_list.append(["ARG"]+ tem+[i.predicate.symbol])
                arg.append(tem+[i.predicate.symbol]+[l])
    else:
            tem = []
            for j in range(0,len(formula.terms)):
                try:
                    tem.append(formula.terms[j].symbol)
                except:
                    tem.append(formula.terms[j].value)
            
            and_list.append(["ARG"]+ tem+[formula.predicate.symbol])
            arg.append(tem+[formula.predicate.symbol]+[l])                                
    return and_list,arg
    

# ===== notebook cell 8 =====

# ===== notebook cell 10 =====
def pysat_formula(formula):
    tem_list = []
    for i in str(formula).split(" & "):
        if i[0] == "x":
            tem_list.append([int(i[1:])])
        else:
            tem_tem = []
            for j in i.replace("(","").replace(")","").split(" | "):
                if j[0] == "~":
                    tem_tem.append(int(j[2:])*-1)
                elif j[0] == "x":
                    tem_tem.append(int(j[1:]))
            tem_list.append(tem_tem)
    return tem_list

# ===== notebook cell 13 =====
no_ = []
def prove(data, threshold,c_threshold):
    # Initialize lists and dictionaries
    for_expressions = []
    check_args = []
    
    replace_x = {}
    n = 1
    if len(data) !=2:
        tem_d = [data[0]]+data[2:]
    else:
        tem_d = [data[0],]
    # Extract relevant parts from data
    for item in tem_d:
        temp_for, temp_check_arg = extract(item)
        for_expressions.append(temp_for)
        check_args.append(temp_check_arg)
    
    # Map arguments to symbols
    for args in check_args:
        for arg in args:
            key = "/".join(arg[:-1])
            if key not in replace_x:
                replace_x[key] = Symbol(f'x{n}')  
                n += 1
    
    check_args_main = []
    # Extract main arguments
    for_expr, check_args_main = extract(data[1])
    
    replace_xx = {}
    max_dict = {}
    comp_dict = {}

    # Initialize max_dict with keys from check_args_main
    for arg in check_args_main:
        key = "/".join(arg[:-1])
        max_dict[key] = 0
        
    
    
    template = {
        ":purpose": "[T1] for [T2]",
    ":time": "[T2] is when action [T1] takes place",
    ":ARG0": "[T2] [T1]",
    ":ARG1": "[T1] [T2]",
    ":direction": "[T2] indicates the direction in action [T1]",
    ":domain": "[T2] is [T1]",
    ":mod": "[T2] modifies action [T1]",
    ":manner": "[T2] describes how action [T1] is executed",
    ":poss": "[T2] possesses [T1]",
    ":poss-of": "[T2] is possessed by [T1]",
    ":topic": "[T2] is the topic of action [T1]",
    ":quant": "[T2] quantifies aspects of action [T1]",
    ":part": "[T2] is part of [T1]",
    ":part-of": "[T2] is part of [T1]",
        ":consist": "[T2] consists of [T1]",
    ":consist-of": "[T1] consists of [T2]",
    ":location": "[T2] is the location of action [T1]",
    ":location-of": "[T2] is the location related to [T1]",
    ":name": "[T1] is named [T2]",
    ":dayperiod": "[T2] is the day period of action [T1]",
    ":destination": "[T2] is the destination of action [T1]",
    ":instrument": "[T2] is the instrument used in action [T1]",
    ":path": "[T2] is the route of action [T1]",
    ":subevent-of": "[T2] is a sub-event of [T1]",
#     ":op1": "[T2] is the first operand in [T1]",
#     ":op2": "[T2] is the second operand in [T1]",
        ":prep-on" : "[T1] on [T2]",
        "none": "[T1] [T2]"}

    # Process main arguments and perform replacements
    for arg in check_args_main:
        key = "/".join(arg[:-1])
        if key in replace_xx:
            continue
        
        if key in replace_x:
            replace_xx[key] = replace_x[key]
            max_dict[key] = 1
            comp_dict[key] = [[1,replace_x[key]]]
        else:
            # Iterate over replace_x to find replacements
            for j_key in replace_x:
                # Check if the argument type exists in the template
                arg_type = arg[-2]
                if arg_type in template:

                    temp_s1 = template[arg_type].replace("[T1]", arg[0]).replace("[T2]", arg[-3])
                elif arg_type[:3] == ":op":
                    
                    if arg[0] == "and":
                        temp_s1 = arg[-3]
                    else:
                        temp_s1 = template["none"].replace("[T1]", arg[0]).replace("[T2]", arg[-3])
                else:
                    temp_s1 = template["none"].replace("[T1]", arg[0]).replace("[T2]", arg[-3])
                    no_.append(arg_type)
                # Process each key in replace_x
                if j_key.split("/")[-1] in template:
                    j_type = j_key.split("/")[-1]
                    temp_s2 = template[j_type].replace("[T1]", j_key.split("/")[0]).replace("[T2]", j_key.split("/")[1])
                elif j_key.split("/")[-1][:3] == ":op":
                    if j_key.split("/")[0] == "and":
                        temp_s2 = j_key.split("/")[1]
                    else:
                        temp_s2 = template["none"].replace("[T1]", j_key.split("/")[0]).replace("[T2]", j_key.split("/")[1])
                        
                else:
                    temp_s2 = template["none"].replace("[T1]", j_key.split("/")[0]).replace("[T2]", j_key.split("/")[1])
                    if j_key.split()[-1] not in no_:
                        no_.append(j_key)
                
                # Substitute expressions (assuming 'subsitute' is a defined function)
                
                subsitute(
                    temp_s2,
                    temp_s1,
                    replace_x,
                    replace_xx,
                    max_dict,
                    key,
                    j_key,
                    threshold,
                    c_threshold,
                    comp_dict,
                    True
                )
            
            # Assign a new symbol if no replacement was found
            if key not in replace_xx:
                replace_xx[key] = Symbol(f'x{n}')
                n += 1
                continue
    

    tem_formula = True
    
    for expr in for_expressions:
        tem_formula &= combine(transform(expr, replace_x))
    
    for i in comp_dict:
        comp_dict[i] = sorted(comp_dict[i], key=itemgetter(-2),reverse=True)
#     for i in replace_xx:
#         if ~replace_xx[i] in replace_xx.values():
#                 all_match = [k for k,v in replace_xx.items() if v == ~replace_xx[i]]
#                 check_max = True
#                 for j in all_match:
#                     if comp_dict[i][0][-2]<comp_dict[j][0][-2]:
#                         check_max = False
#                 if check_max:
#                     new_replace_xx[i] = replace_xx[i]
#                 else:
#                     if len(comp_dict[i])>1:
#                         new_replace_xx[i] = comp_dict[i][1][-1]
#                     else:
#                         new_replace_xx[i] = True
#         else:
#                 new_replace_xx[i] = replace_xx[i]
    formula_main = combine(transform(for_expr, replace_xx))
    
    # Define final formulas
    final_formula = to_cnf(tem_formula & ~formula_main)
    final_formula11 = to_cnf(tem_formula & formula_main)
    # Convert to CNF for SAT solver
    cnf_ent = CNF(from_clauses=pysat_formula(final_formula))
    cnf_con1 = CNF(from_clauses=pysat_formula(final_formula11))
 
    
    check_incon_claim = CNF(from_clauses=pysat_formula(to_cnf(formula_main)))
    with Solver(name="Minisat22", bootstrap_with=check_incon_claim) as solver_check_claim:
        check_claim = solver_check_claim.solve()
    # Solve using SAT solver
    with Solver(name="Minisat22", bootstrap_with=cnf_ent) as solver_ent:
        check_ent = solver_ent.solve()
    
    with Solver(name="Minisat22", bootstrap_with=cnf_con1) as solver_con1:
        check_con1 = solver_con1.solve()
    if not check_claim:
        return False
    # Determine the result based on solver outputs
    if not check_ent and check_con1:
        return "ent", max_dict
    elif not check_con1 and check_ent:
#         for i in replace_x:
#             if "possible" in i:
#                 return "neu", max_dict
                
        return "con", max_dict
    elif not check_con1 and not check_ent:
        return "both", max_dict
    else:
        return "neu", max_dict

# ===== notebook cell 16 =====
def subsitute(x,y,replaceX,replaceXX,maxx,i,j,thre,ct,m = None,mm = False,neg =False):
    
    tems = score(x,y)
#     if temsn>0.5:
    finals = tems
#     finals = tems
    
    if finals >= thre:
        ans = NLI(x,y,nli_tokenizer,model_nli)
        if ans[0] == "contradiction" and ans[1]>=ct:
            if mm:
                if i not in m:
                    m[i] = [[y,j,x,finals,~replaceX[j]]]
                else:
#                   
                    m[i].append([y,j,x,finals,~replaceX[j]])
            
            if finals > maxx[i]:
            
                maxx[i] = finals
                replaceXX[i] = ~replaceX[j]
                return True
#             elif ans[0] == "neutral" and ans[1]>=80:
#                  if finals > maxx[i]:
#                     maxx[i] = finals
                    
        else:
#             if tems>=0.8:
            if mm:
                if i not in m:
                    m[i] = [[y,j,x,finals,replaceX[j]]]
                else:
#                     if tems>
                    m[i].append([y,j,x,finals,replaceX[j]])
            if finals > maxx[i]:
                maxx[i] = finals
                replaceXX[i] = replaceX[j]
                
                return True
            
    if finals > maxx[i]:
        maxx[i] = finals
                    
    return False
            

# ===== notebook cell 17 =====
def transform_logic(x):
    return (remove_duplicate_predicates(((merge_quant_predicates(
                                                       (merge_mod_predicates(
                                                                             (transform_replace_constants(x)))))))))
