import argparse

p = argparse.ArgumentParser()
p.add_argument('--on', action='store_true')
args = p.parse_args(['--on'])
print(args)

args = p.parse_args([])
print(args)