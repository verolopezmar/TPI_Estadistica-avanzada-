## Qué cambia
<!-- Rebanada vertical / historia de usuario que cierra. Ej.: Closes #12 -->

## Fase y rol
- Fase: <!-- 0 a 5 -->
- Autor/a (rol Scrum):

## Evidencia
- [ ] `ruff check .` sin errores
- [ ] `ruff format --check .` sin cambios
- [ ] `pytest -m leakage` en verde (no-fuga Train/Test)
- [ ] `pytest` en verde (pegar resumen abajo)
- [ ] Sin datos crudos ni credenciales en el diff
- [ ] Si toca el notebook: "Restart & Run All" sin errores
- [ ] Si toca la app: `docker build` y prueba de los 3 módulos
- [ ] Ninguna cifra escrita a mano: informe, slides y README leen `models/resultados.json`
- [ ] Si hubo uso de IA: entrada agregada en `docs/ia_log.md`

```
<!-- salida de pytest -->
```

## Veredicto QA (QA & Validation Engineer)
<!-- [APROBADO] o [BLOQUEADO] + motivo. Sin este veredicto el Project Lead no mergea. -->
