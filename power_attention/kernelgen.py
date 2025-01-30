import inspect
import re
import os
import hashlib
from collections import defaultdict
from jinja2 import Template, Environment, meta
from pathlib import Path

def extract_template_from_docstring(func):
    """Extract template code from function docstring between <kernelgen> tags"""
    docstring = func.__doc__ or ""
    match = re.search(r'<kernelgen>(.*?)</kernelgen>', docstring, re.DOTALL)
    # import pdb; pdb.set_trace()
    if match:
        template_str = match.group(1)
        print(f"\nExtracted template from {func.__name__}:")
        print("=" * 80)
        print(template_str)
        print("=" * 80)
        return template_str
    return None

def get_dependent_functions(func):
    """Get the source code of dependent functions that are defined in the same module"""
    module = inspect.getmodule(func)
    dependent_funcs = {}
    
    # Get all referenced names in the function
    func_source = inspect.getsource(func)
    referenced_names = set(re.findall(r'\b\w+\b', func_source))
    
    # Find dependent functions defined in the same module
    for name in referenced_names:
        if hasattr(module, name):
            obj = getattr(module, name)
            if inspect.isfunction(obj) and obj.__module__ == module.__name__:
                dependent_funcs[name] = "@triton.jit\n" + inspect.getsource(obj)
    
    return dependent_funcs

def get_template_variables(template_str):
    """Extract all variables used in the template"""
    env = Environment()
    try:
        ast = env.parse(template_str)
        variables = meta.find_undeclared_variables(ast)
        print("\nTemplate variables found:", sorted(list(variables)))
        return variables
    except Exception as e:
        print("\nError parsing template:")
        print("-" * 80)
        # Print template with line numbers for easier debugging
        for i, line in enumerate(template_str.split('\n'), 1):
            print(f"{i:3d} | {line}")
        print("-" * 80)
        print(f"Error: {str(e)}")
        raise

def get_generated_file_path(func, config_hash):
    """Generate a unique file path for the rendered template"""
    module_path = Path(inspect.getmodule(func).__file__)
    generated_dir = module_path.parent / "generated"
    generated_dir.mkdir(exist_ok=True)
    
    # Create filename using function name and config hash
    filename = f"{func.__name__}_{config_hash}.py"
    return generated_dir / filename

def get_rendered_dir(func):
    """Get the _rendered directory path for the module"""
    module_path = Path(inspect.getmodule(func).__file__)
    rendered_dir = module_path.parent / "_rendered"
    rendered_dir.mkdir(exist_ok=True)
    return rendered_dir

def compute_config_hash(rendered_code):
    """Compute a deterministic hash of config values"""
    return hashlib.md5(rendered_code.encode()).hexdigest()[:8]

class Condition:
    """A class representing a condition, useful for printing out triton-allowed conditions"""
    def __init__(self, conditions=None):
        if isinstance(conditions, dict):
            self.conditions = [conditions]
        elif isinstance(conditions, Condition):
            self.conditions = conditions.conditions
        elif isinstance(conditions, list):
            assert isinstance(conditions[0], dict) or len(conditions) == 0
            self.conditions = conditions
        elif conditions is None:
            self.conditions = []
        else:
            raise ValueError(f"Invalid conditions type: {type(conditions)}")
    
    def __or__(self, other):
        return Condition(self.conditions + other.conditions)
    
    @staticmethod
    def print_condition(condition: dict):
        """ Given a list of key-value pairs, print out a corresponding binary condition
        """
        if len(condition) == 0:
            return "True"
        elif len(condition) == 1:
            key, val = list(condition.items())[0]
            return f"({key} == {val})"
        else:
            key, val = list(condition.items())[0]
            condition.pop(key)
            res = f"({key} == {val}) and ({Condition.print_condition(condition)})"
            condition[key] = val
            return res
    
    def print(self):
        """ Print out the condition as a triton-allowed condition """
        if len(self.conditions) == 0:
            return "True"
        elif len(self.conditions) == 1:
            return Condition.print_condition(self.conditions[0])
        else:
            c0 = self.conditions[0]
            self.conditions.pop(0)
            res = f"({Condition.print_condition(c0)}) or ({self.print()})"
            self.conditions.insert(0, c0)
            return res
    

def get_function_signature(func):
    """Extract function signature including type hints but excluding decorators"""
    source_lines = inspect.getsource(func).split('\n')
    sig_lines = ["@triton.jit"]
    in_signature = False
    parentheses_count = 0
    
    for line in source_lines:
        stripped = line.strip()
        # Skip decorators
        if stripped.startswith('@'):
            continue
            
        # Start of function definition
        if stripped.startswith('def '):
            in_signature = True
            parentheses_count = line.count('(') - line.count(')')
            sig_lines.append(line)
            if parentheses_count == 0:
                break
            continue
            
        # Continue capturing multi-line signature
        if in_signature:
            sig_lines.append(line)
            parentheses_count += line.count('(') - line.count(')')
            if parentheses_count == 0:
                break
    
    return '\n'.join(sig_lines)

def extract_constexpr_declarations(template_str):
    """Extract all constexpr declarations from the template"""
    # Find all lines that contains ": tl.constexpr"
    lines = template_str.split('\n')
    constexpr_lines = [line for line in lines if ": tl.constexpr" in line or ":tl.constexpr" in line]
    rest_lines = [line for line in lines if line not in constexpr_lines]
    return constexpr_lines, "\n".join(rest_lines)

def render_template(template_str, context, func_signature):
    """Render template with given context and proper function definition"""
    # Create the full template with imports and function definition
    # import pdb; pdb.set_trace()
    constexpr_lines, rest_lines = extract_constexpr_declarations(template_str)
    template = Template(rest_lines)
    return template.render(**context), constexpr_lines

def kernelgen(configs):
    def decorator(func):
        if os.environ.get("KERNELGEN", "0") != "1":
            return func
        template_str = extract_template_from_docstring(func)
        if not template_str:
            raise ValueError("No template found in function docstring")
        
        # Get template variables and dependent functions
        template_vars = get_template_variables(template_str)
        dependent_funcs = get_dependent_functions(func)
        rendered_dir = get_rendered_dir(func)
        
        # Generate all variants up front
        variants = defaultdict(set) # dict of rendered_code -> match_dict
        
        print(f"\nPre-rendering kernel variants for {func.__name__}:")
        for config in configs:
            # Create context for this config
            context = {
                var: config.kwargs[var]
                for var in template_vars 
                if var in config.kwargs
            }
            
            # Check for missing variables
            missing_vars = template_vars - set(context.keys())
            if missing_vars:
                print(f"Warning: Config {config} missing variables: {missing_vars}")
                continue
                
            # Render template
            rendered_code, constexpr_lines = render_template(template_str, context, "")  # Empty signature since we'll add it later
            
            # Create match dict for this config
            match_dict = frozenset({
                key: val for key, val in config.kwargs.items()
                if isinstance(val, (int, float, bool, str))  # Only include static values
            }.items())
            
            variants[rendered_code].add(match_dict)
        
        # Generate the dispatcher function code
        dispatcher_code = []

        # Add imports
        dispatcher_code.append("import triton")
        dispatcher_code.append("import triton.language as tl")
        dispatcher_code.append("")

        # Add dependent functions
        for func_code in dependent_funcs.values():
            dispatcher_code.append(func_code)
        
        # Add the main function signature and body
        dispatcher_code.append(get_function_signature(func))
        
        # Add constexpr declarations
        dispatcher_code.extend(f"    {line}" for line in constexpr_lines)
        
        # Add the if-else chain
        first = True
        for rendered_code, match_dicts in variants.items():
            condition = Condition()
            
            for match_dict in match_dicts:
                condition |= Condition(dict(match_dict))
            
            if first:
                dispatcher_code.append(f"    if {condition.print()}:")
                first = False
            else:
                dispatcher_code.append(f"    elif {condition.print()}:")
            
            # Indent the rendered code
            indented_code = "\n".join("        " + line for line in rendered_code.split("\n"))
            dispatcher_code.append(indented_code)
        
        # Add final else clause
        dispatcher_code.append("    else:")
        dispatcher_code.append('        tl.static_assert(False, "No matching config found")')
        
        # Create the final function
        final_code = "\n".join(dispatcher_code)
        
        # Save the generated code to _rendered directory
        file_path = rendered_dir / f"{func.__name__}_dispatcher.py"
        with open(file_path, 'w') as f:
            f.write(final_code)
        print(f"Generated dispatcher -> {file_path}")
        

        # Unfortunately triton will look at the generated code from disk, so we are 
        # forced to have a second run to use the generated code.
        # Create namespace and execute the code
        namespace = {}
        exec(final_code, func.__globals__, namespace)
        
        # Get the generated function
        generated_func = namespace[func.__name__]
        
        # Add source code inspection capabilities
        generated_func.__source__ = final_code
        
        # Override __str__ to show the source code
        def __str__(self):
            return self.__source__
        generated_func.__str__ = __str__.__get__(generated_func)
        
        # Add repr to also show source
        def __repr__(self):
            return f"<Generated function {func.__name__}>\n{self.__source__}"
        generated_func.__repr__ = __repr__.__get__(generated_func)
        
        return generated_func
    
    return decorator