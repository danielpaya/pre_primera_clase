# Justificación de selección de features — NovaMarket

## Proceso y criterios

Partí de las 22 candidatas numéricas de `S13/S13_dataset_con_features.csv`. Separé entrenamiento y prueba por `id_cliente` antes de imputar, calcular correlaciones o ajustar selectores: así ningún cliente aparece en ambas particiones. La partición contiene 441.446 filas de entrenamiento y 110.344 de prueba; las clases representan, respectivamente, 20,25% y 20,55% de las filas. El conjunto de prueba no intervino en la selección.

El filtro eliminó constantes o casi constantes (ninguna candidata cumplió el umbral) y ordenó por información mutua estimada en una muestra reproducible del entrenamiento. El wrapper aplicó RFE con regresión logística balanceada y el método embedded ordenó importancias de Random Forest. Los ajustes de los modelos usaron muestras del entrenamiento. Para la selección final prioricé features coincidentes en más de un método; completé ocho con las de mayor información mutua, y rechacé cualquier candidata correlacionada en más de 0,85 con una feature ya seleccionada.

## Features seleccionadas

| Feature | Métodos que la seleccionaron | Razón específica |
|---|---|---|
| `monto_compra` | Filtro, RFE y Random Forest (3/3) | Captura el valor del pedido actual y fue la señal de mayor importancia del bosque (0,1483); ofrece una medida directa del valor asociado a la compra. |
| `puntos_fidelidad_actuales` | Filtro, RFE y Random Forest (3/3) | Resume la participación del cliente en el programa de fidelidad; MI de entrenamiento 0,004246 y presencia consistente en los tres métodos. Se debe disponer de este dato al momento de puntuar. |
| `monto_por_unidad` | Filtro, RFE y Random Forest (3/3) | Normaliza el gasto por cantidad adquirida y permite distinguir tamaño del pedido de valor unitario; fue consistente en los tres métodos. Se retuvo frente a `precio_unitario` por su mayor coincidencia (3/3). |
| `unidades_vendidas` | Filtro y RFE (2/3) | Describe el volumen de la compra y complementa el importe total; ambos métodos la seleccionaron. |
| `ticket_promedio_previo_cliente` | Filtro y Random Forest (2/3) | Representa el gasto típico histórico sin incluir el pedido actual, según el catálogo S13. Aporta contexto de valor previo y fue respaldada por dos métodos. |
| `edad_cliente` | RFE y Random Forest (2/3) | Aporta una característica demográfica que los dos métodos basados en modelo conservaron; el bosque puede capturar relaciones no lineales que la información mutua univariada no reflejó (MI 0,000846). |
| `antiguedad_cliente_dias` | Filtro (1/3) | Mide el tiempo de relación con el club. Tuvo la MI más alta del filtro (0,006389); se agregó tras las coincidencias de dos o más métodos para completar una selección de ocho. |
| `dias_desde_compra_previa` | Filtro (1/3) | Es una medida de recencia construida con compras de días anteriores, conforme al catálogo S13. Su MI fue 0,002595; se agregó después de priorizar las coincidencias para llegar a ocho. |

La correlación absoluta máxima entre las ocho seleccionadas fue **0,6832**, menor que el límite de 0,85. No hay pares redundantes por el criterio de la actividad.

## Features descartadas

| Feature | Evidencia del entrenamiento | Razón específica para descartarla |
|---|---|---|
| `precio_unitario` | RFE y Random Forest (2/3); MI 0,002592; correlación absoluta 0,9944 con `monto_por_unidad` | Es casi duplicada de `monto_por_unidad` y supera el umbral de redundancia. Se conserva esta última, que coincidió en los tres métodos. |
| `longitud_comentario` | Solo filtro; MI 0,002244; RFE rango 14 | No obtuvo respaldo de RFE ni entró en las ocho importancias principales del bosque. Además, la longitud de un comentario puede no estar disponible en el momento de predecir la fuga. |
| `mes_pedido` | Solo filtro; MI 0,001977 | Su señal temporal estacional no fue confirmada por RFE ni por Random Forest; la prioridad recayó en señales de cliente y compra. |
| `pedidos_previos_cliente` | Solo filtro; MI 0,001071; RFE rango 6 | Mide frecuencia histórica, pero no entró en los ocho elegidos por RFE ni por Random Forest. Para esta selección, el gasto típico previo y la recencia aportan contexto histórico con mayor apoyo metodológico. |
| `peso_pedido_kg` | Solo Random Forest; importancia 0,063336; MI 0,000830 | El bosque la consideró informativa, pero no apareció en filtro ni RFE. El peso describe principalmente logística del pedido y no se priorizó sobre señales de cliente con más coincidencia entre métodos. |
| `es_venta_corporativa` | Solo RFE; MI 0,000341; importancia RF 0,002627 | La regresión logística la conservó, pero el filtro y el bosque no la seleccionaron. La condición corporativa es una señal de tipo de transacción, con apoyo limitado en este análisis. |
| `stock_minimo` | Solo RFE; MI 0,000003; importancia RF 0,016956 | No fue seleccionada por filtro ni por Random Forest. Es una variable operativa de abastecimiento, menos directamente ligada al comportamiento de retención del cliente. |
| `gasto_previo_cliente` | Solo Random Forest; importancia 0,061406; MI 0,000000 | El bosque la incluyó, pero filtro y RFE no. El total acumulado puede crecer con el número de pedidos previos; se priorizó `ticket_promedio_previo_cliente`, que expresa gasto típico y tuvo coincidencia en dos métodos. |
| `num_palabras_comentario` | Ningún método; MI 0,000021; RFE rango 2 | No alcanzó la selección de ocho de RFE ni apareció en filtro o bosque. También mide longitud de texto y añade poco frente a `longitud_comentario`, que tampoco llegó a la selección final. |
| `rfm_score` | Ningún método; MI 0,000011; RFE rango 10 | El score combina umbrales de recencia, frecuencia y valor. No fue seleccionado y su granularidad es menor que conservar por separado las señales continuas con respaldo metodológico. |
| `es_fin_semana_pedido` | Ningún método; MI 0,000008; RFE rango 15 | El indicador de fin de semana tuvo señal univariada mínima y no fue conservado por ninguno de los tres métodos. |
| `menciona_demora` | Ningún método; MI 0,000007; RFE rango 3 | No llegó a las ocho variables de RFE y su importancia en Random Forest fue 0,003137. Además, el comentario que menciona una demora puede conocerse después de la compra, por lo que su disponibilidad temporal es dudosa. |
| `tiene_comentario` | Ningún método; MI 0,000004; RFE rango 11 | La mera presencia de un comentario tuvo señal mínima y no fue elegida por ningún método; no añade información suficiente frente a características de compra e historial. |
| `dias_reposicion` | Ningún método; MI 0,000004; RFE rango 4 | Aunque su importancia RF fue 0,017619, no estuvo entre las ocho principales del bosque y RFE tampoco la conservó. Describe reposición de inventario, no una interacción directa del cliente. |

## Control de calidad y uso posterior

El CSV final conserva **551.790 filas** y contiene exactamente estas columnas, en este orden: `monto_compra`, `puntos_fidelidad_actuales`, `monto_por_unidad`, `unidades_vendidas`, `ticket_promedio_previo_cliente`, `edad_cliente`, `antiguedad_cliente_dias`, `dias_desde_compra_previa`, `fuga_cliente` e `id_cliente`.

**Advertencia sobre la etiqueta:** el dataset contiene 74.938 clientes con ambas clases de `fuga_cliente` entre sus pedidos. La partición agrupada evita que el mismo cliente quede en entrenamiento y prueba, pero no corrige esta inconsistencia. Antes de entrenar el modelo de fuga, conviene confirmar si la etiqueta está definida por pedido o por cliente y resolverla si se esperaba una única clase por cliente. No se modificaron las etiquetas originales.

Como en cualquier predicción, confirma que cada feature elegida esté disponible en el instante real de scoring; en particular, `puntos_fidelidad_actuales` debe ser una captura vigente en ese momento. La selección de features no reemplaza la evaluación predictiva ni el tratamiento de desbalance previsto para la Semana 15.
