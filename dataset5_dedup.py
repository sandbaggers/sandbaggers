"""Conservative C/C++ token clone keys; preserve literals and token boundaries."""
import json
import re

# Keep maximal-munch operators distinct from whitespace-separated operators.
TOKEN = re.compile(
    r'(?P<space>\s+)|(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)|'
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\('
    r'[\s\S]*?\)(?P=delimiter)")|'
    r'(?P<literal>(?:u8|u|U|L)?(?:"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'))|'
    r'(?P<word>[A-Za-z_$][A-Za-z_0-9$]*|[0-9][A-Za-z_0-9.]*)|'
    r'(?P<operator><=>|>>=|<<=|->\*|\.\.\.|::|->|\.\*|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||'
    r'\+=|-=|\*=|/=|%=|\^=|&=|\|=|##)|(?P<other>[\s\S])')


def body_key(row):
    tokens = [m.group() for m in TOKEN.finditer(row['code'])
              if m.lastgroup not in ('space', 'comment')]
    # Ignore extraction differences in return/storage declarations. This is
    # conservative clone removal, not a claim of general semantic equivalence.
    if '{' in tokens:
        tokens = tokens[tokens.index('{'):]
    return json.dumps(tokens, ensure_ascii=False, separators=(',', ':'))
