import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
  name: "PreviewText",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "PreviewText") {
      function populate(text) {
        if (this.widgets) {
          for (let i = 1; i < this.widgets.length; i++) {
            this.widgets[i].onRemove?.();
          }
          this.widgets.length = 1;
        }

        const widget = ComfyWidgets["STRING"](
          this,
          "text",
          ["STRING", { multiline: true }],
          app
        ).widget;
        widget.inputEl.readOnly = true;
        widget.value = text.join("");
        widget.inputEl.style.backgroundColor = "rgba(0,0,0,0)"

        requestAnimationFrame(() => {
          const size = this.computeSize();
          if (size[0] < this.size[0]) {
            size[0] = this.size[0];
          }
          if (size[1] < this.size[1]) {
            size[1] = this.size[1];
          }
          this.onResize?.(size);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        populate.call(this, message.text);
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        if (this.widgets_values?.length) {
          populate.call(this, this.widgets_values);
        }
      };
    }
  },
});
