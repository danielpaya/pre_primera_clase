# Decisiones de reducción y balanceo

## División, datos faltantes y prevención de fuga

Se separó el 20% para prueba mediante `train_test_split` con `stratify=df["fuga_cliente"]` y `random_state=42`. Entrenamiento: clase 0 = 351.784 y clase 1 = 89.648. Prueba: clase 0 = 87.946 y clase 1 = 22.412; la prevalencia de fuga en prueba es 20,31%. El CSV de entrada tiene una prevalencia positiva de 20,31%, no el ~9% esperado; la estratificación conserva la distribución observada sin alterar las etiquetas.

El identificador `id_cliente` se excluyó de las features predictoras. Los infinitos en features se trataron como faltantes y se imputaron con la mediana, ajustada solo en entrenamiento. `SimpleImputer` y `StandardScaler` se ajustaron exclusivamente con `X_train`; los mismos ajustes transformaron la prueba. El conjunto de prueba no se usó para ajustar imputación, escalado, PCA ni técnicas de balanceo.

## Correlación y PCA

La matriz de correlación se calculó como `X_train.corr()`. Criterio: usar PCA si al menos un par tiene `|correlación| > 0.6`.

Se aplicó PCA porque se detectaron correlaciones absolutas mayores a 0.6 (`monto_compra`–`unidades_vendidas`: 0,684; `monto_compra`–`monto_por_unidad`: 0,613). Se conservaron 7 componentes, que explican 99,0% de la varianza. PCA se ajustó solo con entrenamiento.

PCA es una transformación lineal que combina features y puede reducir redundancia y colinealidad. Se usó `n_components=0.90` para conservar al menos el 90% de la varianza; el ajuste se hizo solo con entrenamiento y se aplicó sin reajustes a la prueba.

## Comparación del balanceo

Las tres técnicas se aplicaron solo a entrenamiento con `random_state=42`:

| Técnica | Filas | Clase 0 | Clase 1 | Features duplicadas |
| --- | ---: | ---: | ---: | ---: |
| RandomUnderSampler | 179.296 | 89.648 | 89.648 | 869 |
| RandomOverSampler | 703.568 | 351.784 | 351.784 | 265.244 |
| SMOTE | 703.568 | 351.784 | 351.784 | 3.702 |

- **RandomUnderSampler:** reduce la clase mayoritaria, pero puede descartar información útil.
- **RandomOverSampler:** replica casos de la minoritaria y conserva los datos originales, aunque puede aumentar el sobreajuste por duplicación.
- **SMOTE:** crea ejemplos sintéticos de la minoritaria a partir de vecinos cercanos, manteniendo la mayoría y evitando duplicados exactos. Puede crear combinaciones poco realistas; sus resultados deben validarse con métricas apropiadas para clases desbalanceadas.

Se eligió **SMOTE** para el entrenamiento final por el balance entre retener los datos mayoritarios y diversificar la clase minoritaria. La prueba se mantuvo sin balancear para evaluar posteriormente con su prevalencia real.

## Archivos

- `novamarket_train_balanceado.csv`: features finales de entrenamiento y etiqueta `fuga_cliente`; `id_cliente` queda excluido.
- `novamarket_test.csv`: filas de prueba originales, con todas sus columnas y valores, sin escalar ni balancear.
