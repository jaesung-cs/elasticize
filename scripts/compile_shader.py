import sys
import os
import glob
import functools
import operator

if __name__ == "__main__":
  if len(sys.argv) >= 2:
    dirpath = sys.argv[1]
  else:
    dirpath = os.path.dirname(sys.argv[0])

  extensions = ['vert', 'frag', 'geom', 'tesc', 'tese', 'comp']
  filenames = functools.reduce(operator.add, [[os.path.abspath(path) for path in glob.glob(f'{dirpath}/**/*.{extension}', recursive = True)] for extension in extensions])

  for filename in filenames:
    # compare change date
    target_filename = f'{filename}.spv'

    source_date = os.path.getmtime(filename)
    target_date = os.path.getmtime(target_filename) if os.path.exists(target_filename) else 0.

    if source_date > target_date:
      print(f'compiling {filename}:')
      if os.system(f'glslc.exe {filename} -o {target_filename}') != 0:
        # delete previously compiled spv file
        print(f'failed to compile shader: {filename}')
        if os.path.exists(f'{filename}.spv'):
          os.remove(f'{filename}.spv')
