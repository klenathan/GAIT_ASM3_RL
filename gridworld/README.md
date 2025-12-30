# Train commands

```
python -m main --level 4 --episodes 100000 --save_model level4_q_sarsa.pkl --no_render --algo=sarsa
```

# Test commands

```
python main.py --level 4 --load_model level4_q.pkl --test --render_delay 0.3
```