# Reator flexível Batch/CSTR

O modelo foi organizado no pacote `reator_flex`, mantendo a rede:

`A + B <-> C -> I -> P`

## Estrutura

- `model.py`: parâmetros, Arrhenius, velocidades e balanços diferenciais.
- `simulation.py`: integração com `solve_ivp`/`BDF`.
- `reporting.py`: métricas e tabelas do estado final.
- `plotting.py`: gráficos.
- `main.py`: execução de um caso CSTR padrão.

## Como executar

Na raiz do repositório:

```bash
python -m reator_flex.main
```

As dependências são `numpy`, `scipy`, `pandas` e `matplotlib`.

O arquivo original foi mantido na raiz para preservar compatibilidade. A versão organizada e recomendada fica nesta pasta.
