# convert json to yaml
# http://pyyaml.org/wiki/PyYAMLDocumentation
# python3 json2yaml.py < ~/code/manpow/moneybug/mbuploader/support/offices.json
# gist https://gist.github.com/noahcoad/46909253a5891af3699580b8f17baba8

import yaml, json, sys
sys.stdout.write(yaml.dump(json.load(sys.stdin)))
