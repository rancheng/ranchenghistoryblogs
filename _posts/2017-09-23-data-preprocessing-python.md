---
published: false
---
A simple script to remove redundent white space in a text file:

```python
import re
with open(fname, 'r') as input_f, open(fname_new, 'w') as output_f:
	for line in input_f:
    	line_out = re.sub(' +', ' ', line)
        output_file.write(line_out)
```

You can customize anything using this script, with regular expression.