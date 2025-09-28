# SimMobileResilience — стенд

Еталонна імплементація моделі міжрівневих порушень і інтегрального показника **B\*** для інформаційних систем на мобільній платформі. Репозиторій містить мінімально достатній код для відтворення експериментів, генерації метрик і фігур.

## Встановлення

```bash
python -m venv .venv && source .venv/bin/activate  # або еквівалент у Windows
pip install -r requirements.txt
```

## Запуск (реплікація результатів)

```bash
python sim_mobile_resilience.py --policy both --T 420 --seed 7 --outdir out
```

Отримаєте:
- `out/sim_results_reactive.csv`, `out/sim_results_myopic.csv` — часові ряди;
- `out/sim_summary_*.json`, `out/sim_summary_all.json` — агреговані метрики;
- фігури: `fig_Bstar_time.png`, `fig_blocks_bar.png`, `fig_slo_time_*.png`, `fig_contracts_time_*.png`, `fig_Bstar_hist.png`.

> **Нотатка про графіки.** Скрипт навмисне **не задає кольорів** і **не будує субплоти**.

## Налаштування параметрів

Редагуйте `params.yaml` і передавайте шлях у CLI:

```bash
python sim_mobile_resilience.py --policy myopic --T 600 --seed 123 --outdir out_exp --params params.yaml
```

У YAML зібрані пороги інваріантів, ваги блоків **B\***, радіус локалізації `r_star`, параметри CVaR, частоти «вікон доступності» тощо.

## Структура моделі

- **Локальні інваріанти:** `local_violations` (див. 4.2.1).
- **Переривчаста зв’язність:** випадкові маски `alpha`, генератор `sample_env`.
- **Тропічне поширення:** `tropical_path_contrib` (max–times), радіус `r_star=2`.
- **Контрактація:** `contract_residuals` (масковані залишки).
- **Інтегральний показник:** `compute_Bstar` (структура + контракти + ризик (CVaR)).
- **Політики:** `policy_reactive`, `policy_myopic`.
- **Симуляція:** `run_sim`, графіки у `out/`.
