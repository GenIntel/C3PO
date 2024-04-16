## Contributing

### Feature Development

```
git checkout -b branch-feature
...
git add ...
git commit ...
...
git rebase main
git push --force-with-lease # or even git push -f
# note: this is allowed because rebased branch cannot be pushed
```


### [Optional] Pre-Commit
```
pip install pre-commit
pre-commit run --all-files
```
