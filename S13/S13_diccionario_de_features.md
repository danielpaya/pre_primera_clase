# Diccionario de features: NovaMarket S13

**Objetivo:** aportar señales de retención y valor por cliente a partir del dataset integrado de S12. La unidad de salida sigue siendo un pedido. Las agregaciones de historial para cada fila solo consideran compras de fechas anteriores; todas las compras del mismo cliente y día se agrupan antes de calcular el historial.

## Variables derivadas

| Feature | Fórmula | Justificación de negocio |
|---|---|---|
| `antiguedad_cliente_dias` | `fecha_pedido - fecha_inscripcion_club`, en días. Si alguna fecha falta o el resultado es negativo, queda nula. | Distingue clientes recién vinculados de clientes con relación más larga con el programa, útil para segmentar retención y analizar el valor según antigüedad. |
| `monto_por_unidad` | `monto_compra / unidades_vendidas`, solo cuando las unidades son mayores que cero. | Resume el gasto unitario del pedido y permite comparar el valor de compras de distinto tamaño. |

## Agregaciones históricas por cliente

Todas se calculan agrupando primero por `id_cliente` y `fecha_pedido`. El día actual se excluye: su importe y su conteo se restan del acumulado, y la fecha previa se obtiene del día distinto anterior. Esto evita que pedidos del mismo día se filtren entre sí.

| Feature | Fórmula en el momento del pedido | Justificación de negocio |
|---|---|---|
| `pedidos_previos_cliente` | Conteo acumulado de pedidos únicos del cliente en fechas anteriores. | Mide la frecuencia histórica de compra, señal directa de recurrencia. |
| `gasto_previo_cliente` | Suma de `monto_compra` del cliente en fechas anteriores. | Aproxima el valor monetario acumulado conocido hasta ese momento, sin incorporar el pedido que se está evaluando. |
| `ticket_promedio_previo_cliente` | `gasto_previo_cliente / pedidos_previos_cliente`; nulo cuando no hay pedidos previos. | Distingue clientes con frecuencia similar pero distinto gasto típico. |
| `dias_desde_compra_previa` | Días entre `fecha_pedido` y la fecha distinta de compra inmediatamente anterior; nulo en el primer día observado del cliente. | Mide recencia, una señal útil para detectar enfriamiento de la relación. |

## Feature de dominio

| Feature | Fórmula | Justificación de negocio |
|---|---|---|
| `rfm_score` | Suma de tres puntuaciones de 1 a 3: **R** vale 3 si no hay compra previa o han pasado hasta 30 días, 2 entre 31 y 90 días y 1 después de 90 días; **F** vale 3 con al menos 5 pedidos previos, 2 con 2 a 4 y 1 con 0 a 1; **M** vale 3 con gasto previo de al menos 1.000.000, 2 con 250.000 a 999.999 y 1 por debajo de 250.000. RFM queda nulo si no se puede calcular el historial por falta de cliente o fecha. | Combina recencia, frecuencia y valor monetario en una señal legible para priorizar campañas de retención y explorar segmentos de alto valor o riesgo. El primer pedido observado recibe R=3 como cliente reciente, y F=M=1 por ausencia de historial. |

Los umbrales monetarios están expresados en las unidades monetarias del dataset (aparentemente COP) y son reglas iniciales de negocio, no límites aprendidos estadísticamente. Deben revisarse con el equipo y validarse en datos de entrenamiento; si se calibran con distribuciones, el ajuste debe hacerse solo dentro del conjunto de entrenamiento de cada partición.

## Control de fuga y uso

- Las agregaciones son *as-of*: solo utilizan días anteriores al `fecha_pedido` de cada fila; no incluyen información de pedidos posteriores.
- No se calcula el historial a partir del campo objetivo `fuga_cliente`. Ese campo permanece en el archivo de salida por provenir de S12, pero debe separarse y excluirse de las variables predictoras al entrenar.
- Una fecha de pedido inválida o un `id_cliente` faltante impide construir agregaciones y RFM para esa fila; las features correspondientes quedan nulas y la fila se conserva.
- El notebook exporta `S13_dataset_con_features.csv` en la carpeta `S13` al ejecutarse.
