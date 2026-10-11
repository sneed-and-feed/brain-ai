import json
from brain_ai.eval.arc_scorer import load_kaggle_split, score_submission
from brain_ai.eval.stats import wilson_interval

kdir = r'C:\Users\x\.gemini\antigravity\brain\cb4ee624-4136-4ca8-a2ec-558926a12173\scratch\TRM\kaggle\combined\arc-agi'
_, sols = load_kaggle_split(kdir, 'evaluation')

sub3758 = json.load(open('results/arc_trm_b1/submission_step3758.json'))
rep3758 = score_submission(sub3758, sols)
lo3758, hi3758 = wilson_interval(rep3758.tasks_fully_solved, rep3758.n_tasks)

sub5010 = json.load(open('results/arc_trm_b1/submission_final_step5010.json'))
rep5010 = score_submission(sub5010, sols)
lo5010, hi5010 = wilson_interval(rep5010.tasks_fully_solved, rep5010.n_tasks)

summary = {
    'benchmark': 'ARC-AGI-1 Public Evaluation (N=400)',
    'baseline': 'B1: Tiny Recursive Model (TRM) Pilot',
    'total_steps': 5010,
    'total_epochs': 324,
    'n_parameters': 6829058,
    'gpu': 'NVIDIA A100-SXM4-80GB',
    'checkpoints': {
        'step_3758': {
            'exact_match_score': rep3758.score,
            'tasks_fully_solved': rep3758.tasks_fully_solved,
            'solved_task_ids': [t for t, v in rep3758.per_task.items() if v == 1.0],
            'wilson_95_ci': [lo3758, hi3758],
            'pass_at_100_evaluator': 0.01375,
            'pass_at_100_tasks_solved': 11
        },
        'step_5010_final': {
            'exact_match_score': rep5010.score,
            'tasks_fully_solved': rep5010.tasks_fully_solved,
            'solved_task_ids': [t for t, v in rep5010.per_task.items() if v == 1.0],
            'wilson_95_ci': [lo5010, hi5010],
            'pass_at_5_evaluator': 0.00875,
            'pass_at_5_tasks_solved': 7,
            'pass_at_100_evaluator': 0.01875,
            'pass_at_100_tasks_solved': 15
        }
    }
}

with open('results/arc_trm_b1/b1_eval_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Summary written successfully:')
print(json.dumps(summary, indent=2))
