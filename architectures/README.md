# Architecture diagram sources

The project includes editable Graphviz `.gv` sources and rendered PNG/SVG previews. From PowerShell, with Graphviz installed:

```powershell
dot -Tpng Architecture.gv -o Architecture.png
dot -Tsvg Architecture.gv -o Architecture.svg
dot -Tpng "Mermaid Diagram.gv" -o "Mermaid Diagram.png"
```

The file called `Mermaid Diagram.gv` is a Graphviz editable overview source (not Mermaid syntax); it is named this way to match the requested deliverable filename. The separate `Mermaid Diagram.md` contains the Mermaid flowchart source. Detailed views cover ingestion, news, features, training/evaluation, serving/monitoring, deployment and SageMaker MLOps.


## V14 deployment-scope note

The main V14 diagrams distinguish the resources defined in the current `template.yaml` (S3 + scheduled Lambda collector + IAM + CloudWatch Logs) from the future/separate SageMaker training, Model Registry, endpoint and Model Monitor workflow. Some detailed historical diagrams depict the intended SageMaker MLOps design; read them as conceptual workflow diagrams unless their resources have separately been created by the SageMaker helper scripts. The SAM template alone does not create those SageMaker resources.
