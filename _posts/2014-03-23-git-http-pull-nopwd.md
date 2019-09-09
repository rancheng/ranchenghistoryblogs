---
published: false
layout: post
title: github http pull password free bashscript
---
Input password every time is annoying, here's a tiny script that can save your time:
```sh
#!/bin/bash
cd [your git repo]
git pull "http://username:password@github.com/xxx/project/repo.git" master
```

or you can upgrade it into taking different branches by using arguments:

```sh
#!/bin/bash
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 repo-directory $1 branch-name " >&2
  exit 1
fi
cd $0
git pull "http://username:password@github.com/xxx/project/repo.git" $1
```

translation: if argument number is not equal to 2, print the error message to `stderr` and return. Else take `$0` as the repo directory and pull the branch `$1`.