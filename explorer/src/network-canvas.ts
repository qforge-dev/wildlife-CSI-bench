import * as d3 from "d3";
import { displayName } from "./graph-data";
import type { Edge, GraphView, Network } from "./graph-data";
import type { CameraState } from "./url-state";

type Point = {
  id: number;
  x: number;
  y: number;
  radius: number;
  weight: number;
  group: number;
};
export const palette = [
  "#8dbfa5",
  "#c7a377",
  "#8eaad1",
  "#be9bc3",
  "#bfc88f",
  "#83bdbf",
];
const neutral = "#727c7d";

export class NetworkCanvas {
  private ctx: CanvasRenderingContext2D;
  private width = 800;
  private height = 600;
  private transform = d3.zoomIdentity;
  private positions = new Map<number, Point>();
  private view: GraphView = {
    nodes: [],
    edges: [],
    neighbors: new Map(),
    errors: 0,
    totalErrors: 0,
  };
  private mode = "";
  private selected: number | null = null;
  private selectedEdge: string | null = null;
  private hovered: number | null = null;
  private frame = false;
  private zoom: d3.ZoomBehavior<HTMLCanvasElement, unknown>;
  private resize: ResizeObserver;
  private moved = false;
  private ready = false;
  private pendingCamera: CameraState | null = null;
  private initialFitIds: Set<number> | undefined;

  constructor(
    private canvas: HTMLCanvasElement,
    private data: Network,
    private onNode: (id: number) => void,
    private onEdge: (edge: Edge) => void,
    private onHover: (text: string | null, x: number, y: number) => void,
    private onZoom: (level: number) => void,
    private onCameraChange: (camera: CameraState) => void,
  ) {
    this.ctx = canvas.getContext("2d")!;
    this.zoom = d3
      .zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.05, 20])
      .filter(
        (event) =>
          event.type === "wheel" ||
          event.type === "dblclick" ||
          (!event.button && !this.pickNode(event)),
      )
      .on("zoom", (event) => {
        this.transform = event.transform;
        this.onZoom(event.transform.k);
        this.drawSoon();
      })
      .on("end", () => {
        if (this.ready) this.onCameraChange(this.getCamera());
      });
    const element = d3.select(canvas);
    element.call(this.zoom);
    element.call(
      d3
        .drag<HTMLCanvasElement, unknown, Point>()
        .subject((event) => this.pickNode(event.sourceEvent)!)
        .on("start", () => {
          this.moved = false;
        })
        .on("drag", (event) => {
          const source = event.sourceEvent;
          const [x, y] = d3.pointer(
            source.changedTouches?.[0] || source,
            canvas,
          );
          event.subject.x = this.transform.invertX(x);
          event.subject.y = this.transform.invertY(y);
          this.moved = true;
          this.drawSoon();
        }),
    );
    canvas.addEventListener("click", (event) => {
      if (event.defaultPrevented || this.moved) {
        this.moved = false;
        return;
      }
      const node = this.pickNode(event);
      if (node) {
        this.onNode(node.id);
        return;
      }
      const edge = this.pickEdge(event);
      if (edge) this.onEdge(edge);
    });
    canvas.addEventListener("mousemove", (event) => {
      const node = this.pickNode(event),
        edge = node ? null : this.pickEdge(event);
      this.hovered = node?.id ?? null;
      canvas.style.cursor = node ? "grab" : edge ? "pointer" : "default";
      const [x, y] = d3.pointer(event, canvas);
      this.onHover(
        node
          ? displayName(this.data.nodes[node.id])
          : edge
            ? `${displayName(this.data.nodes[edge.a])} ${this.mode === "direct" ? "→" : "↔"} ${displayName(this.data.nodes[edge.b])}`
            : null,
        x,
        y,
      );
      this.drawSoon();
    });
    canvas.addEventListener("mouseleave", () => {
      this.hovered = null;
      this.onHover(null, 0, 0);
      this.drawSoon();
    });
    this.resize = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      if (!width || !height) return;
      const camera =
        this.pendingCamera || (this.ready ? this.getCamera() : null);
      this.width = width;
      this.height = height;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.ready = true;
      this.pendingCamera = null;
      if (camera) this.setCamera(camera);
      else this.fit(false, this.initialFitIds);
    });
    this.resize.observe(canvas.parentElement!);
  }
  setGraph(
    view: GraphView,
    mode: string,
    selected: number | null,
    edge: string | null,
  ) {
    this.view = view;
    this.selected = selected;
    this.selectedEdge = edge;
    if (mode !== this.mode) {
      this.mode = mode;
      this.positions.clear();
      for (const id of view.nodes) {
        this.addPoint(id);
      }
    }
    for (const id of view.nodes) if (!this.positions.has(id)) this.addPoint(id);
    for (const point of this.positions.values()) point.weight = 0;
    for (const e of view.edges) {
      this.positions.get(e.a)!.weight += e.count || e.similarity;
      this.positions.get(e.b)!.weight += e.count || e.similarity;
    }
    for (const p of this.positions.values())
      p.radius = 2.3 + Math.min(8, Math.sqrt(p.weight) * 0.3);
    this.drawSoon();
  }
  private addPoint(id: number) {
    const p =
      this.data.nodes[id][this.mode === "profile" ? "profile" : "direct"];
    if (p)
      this.positions.set(id, {
        id,
        x: p[0] * 1000,
        y: p[1] * 1000,
        group: p[2],
        weight: 0,
        radius: 3,
      });
  }
  fit(animate = true, ids?: Set<number>) {
    if (!this.ready) this.initialFitIds = ids;
    const points = this.view.nodes
      .filter((id) => !ids || ids.has(id))
      .map((id) => this.positions.get(id)!);
    if (!points.length) {
      this.drawSoon();
      return;
    }
    const [x0, x1] = d3.extent(points, (p) => p.x) as [number, number],
      [y0, y1] = d3.extent(points, (p) => p.y) as [number, number];
    const k = Math.min(
      8,
      0.8 / Math.max((x1 - x0 + 80) / this.width, (y1 - y0 + 80) / this.height),
    );
    const t = d3.zoomIdentity
      .translate(this.width / 2, this.height / 2)
      .scale(k)
      .translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
    const selection = d3.select(this.canvas);
    if (
      animate &&
      !window.matchMedia("(prefers-reduced-motion:reduce)").matches
    )
      selection.transition().duration(300).call(this.zoom.transform, t);
    else selection.call(this.zoom.transform, t);
  }
  zoomBy(factor: number) {
    d3.select(this.canvas)
      .transition()
      .duration(180)
      .call(this.zoom.scaleBy, factor);
  }
  getCamera(): CameraState {
    return {
      x: this.transform.invertX(this.width / 2),
      y: this.transform.invertY(this.height / 2),
      zoom: (this.transform.k * 1000) / Math.min(this.width, this.height),
    };
  }
  setCamera(camera: CameraState) {
    if (!this.ready) {
      this.pendingCamera = camera;
      return;
    }
    const k = Math.max(
      0.05,
      Math.min(20, (camera.zoom * Math.min(this.width, this.height)) / 1000),
    );
    const transform = d3.zoomIdentity
      .translate(this.width / 2, this.height / 2)
      .scale(k)
      .translate(-camera.x, -camera.y);
    d3.select(this.canvas).interrupt().call(this.zoom.transform, transform);
  }
  private screen(p: Point) {
    return [this.transform.applyX(p.x), this.transform.applyY(p.y)];
  }
  private pickNode(event: MouseEvent | TouchEvent) {
    const source = "changedTouches" in event ? event.changedTouches[0] : event;
    const [x, y] = d3.pointer(source, this.canvas);
    let closest: Point | null = null,
      best = Infinity;
    for (const id of this.view.nodes) {
      const n = this.positions.get(id)!,
        [nx, ny] = this.screen(n),
        distance = Math.hypot(nx - x, ny - y);
      if (
        distance <
          Math.max(
            event.type.startsWith("touch") ? 22 : 10,
            n.radius * Math.sqrt(this.transform.k) + 4,
          ) &&
        distance < best
      ) {
        closest = n;
        best = distance;
      }
    }
    return closest;
  }
  private pickEdge(event: MouseEvent) {
    if (this.transform.k < 0.7 && this.selected === null) return null;
    const [x, y] = d3.pointer(event, this.canvas);
    let result: Edge | null = null,
      best = 5;
    for (const e of this.view.edges) {
      if (
        this.selected !== null &&
        e.a !== this.selected &&
        e.b !== this.selected
      )
        continue;
      const [ax, ay] = this.screen(this.positions.get(e.a)!),
        [bx, by] = this.screen(this.positions.get(e.b)!);
      const dx = bx - ax,
        dy = by - ay,
        l = dx * dx + dy * dy;
      if (!l) continue;
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l));
      const distance = Math.hypot(x - ax - t * dx, y - ay - t * dy);
      if (distance < best) {
        best = distance;
        result = e;
      }
    }
    return result;
  }
  private drawSoon() {
    if (this.frame) return;
    this.frame = true;
    requestAnimationFrame(() => {
      this.frame = false;
      this.draw();
    });
  }
  private draw() {
    const c = this.ctx;
    c.clearRect(0, 0, this.width, this.height);
    const selected = this.selected,
      selectedEdge = this.view.edges.find((e) => e.id === this.selectedEdge),
      highlight = selected !== null || !!selectedEdge;
    const close = selectedEdge
      ? new Set([selectedEdge.a, selectedEdge.b])
      : selected === null
        ? null
        : new Set([selected, ...(this.view.neighbors.get(selected) || [])]);
    for (const e of this.view.edges) {
      const a = this.positions.get(e.a)!,
        b = this.positions.get(e.b)!,
        [ax, ay] = this.screen(a),
        [bx, by] = this.screen(b);
      if (
        Math.max(ax, bx) < 0 ||
        Math.min(ax, bx) > this.width ||
        Math.max(ay, by) < 0 ||
        Math.min(ay, by) > this.height
      )
        continue;
      const active = selectedEdge
        ? e.id === selectedEdge.id
        : selected !== null && (e.a === selected || e.b === selected);
      c.strokeStyle = active ? "#d4e3dc" : "#657471";
      c.globalAlpha = active ? 0.85 : highlight ? 0.035 : 0.14;
      c.lineWidth = active
        ? 1 + Math.log1p(e.count || e.similarity) * 0.4
        : 0.45 + Math.log1p(e.count || e.similarity) * 0.12;
      c.beginPath();
      c.moveTo(ax, ay);
      c.lineTo(bx, by);
      c.stroke();
      if (this.mode === "direct" && (active || this.transform.k > 1.5)) {
        const dx = bx - ax,
          dy = by - ay,
          len = Math.hypot(dx, dy),
          r = b.radius * Math.sqrt(this.transform.k) + 3;
        if (len > r + 12) {
          const x = bx - (dx / len) * r,
            y = by - (dy / len) * r;
          c.fillStyle = active ? "#d4e3dc" : "#657471";
          c.beginPath();
          c.moveTo(x, y);
          c.lineTo(
            x - (dx / len) * 6 + (dy / len) * 3,
            y - (dy / len) * 6 - (dx / len) * 3,
          );
          c.lineTo(
            x - (dx / len) * 6 - (dy / len) * 3,
            y - (dy / len) * 6 + (dx / len) * 3,
          );
          c.fill();
        }
      }
    }
    c.globalAlpha = 1;
    const onscreen: { p: Point; x: number; y: number; r: number }[] = [];
    for (const id of this.view.nodes) {
      const p = this.positions.get(id)!,
        [x, y] = this.screen(p),
        r = Math.max(1.6, p.radius * Math.sqrt(this.transform.k));
      if (x < -20 || x > this.width + 20 || y < -20 || y > this.height + 20)
        continue;
      const active =
        id === selected ||
        id === this.hovered ||
        (selectedEdge && (id === selectedEdge.a || id === selectedEdge.b));
      c.globalAlpha = active ? 1 : close && !close.has(id) ? 0.14 : 0.9;
      c.fillStyle = palette[p.group] || neutral;
      c.strokeStyle = c.fillStyle;
      c.lineWidth = 0.8;
      c.beginPath();
      c.arc(x, y, r, 0, 2 * Math.PI);
      if (this.data.nodes[id].support) c.fill();
      else c.stroke();
      if (active) {
        c.globalAlpha = 1;
        c.strokeStyle = "#f3f7f4";
        c.lineWidth = 1.5;
        c.stroke();
      }
      onscreen.push({ p, x, y, r });
    }
    c.globalAlpha = 1;
    c.font = "12px ui-sans-serif, system-ui, sans-serif";
    c.fillStyle = "#dce3de";
    const boxes: { x: number; y: number; w: number; h: number }[] = [];
    onscreen.sort(
      (a, b) =>
        (b.p.id === selected || b.p.id === this.hovered ? 1e9 : b.p.weight) -
        (a.p.id === selected || a.p.id === this.hovered ? 1e9 : a.p.weight),
    );
    const limit = this.transform.k > 2 ? 70 : 16;
    let count = 0;
    for (const { p, x, y, r } of onscreen) {
      if (count >= limit) break;
      if (close && !close.has(p.id) && p.id !== this.hovered) continue;
      const node = this.data.nodes[p.id],
        text = displayName(node),
        w = c.measureText(text).width;
      for (const [tx, ty] of [
        [x + r + 5, y + 4],
        [x - w / 2, y - r - 8],
        [x - w / 2, y + r + 17],
        [x - r - w - 5, y + 4],
      ]) {
        const box = { x: tx, y: ty - 12, w, h: 16 };
        if (
          tx < 8 ||
          tx + w > this.width - 8 ||
          ty < 18 ||
          ty > this.height - 8 ||
          boxes.some(
            (b) =>
              box.x < b.x + b.w + 5 &&
              box.x + box.w > b.x - 5 &&
              box.y < b.y + b.h + 4 &&
              box.y + box.h > b.y - 4,
          )
        )
          continue;
        c.strokeStyle = "#101718";
        c.lineWidth = 3;
        c.strokeText(text, tx, ty);
        c.fillText(text, tx, ty);
        boxes.push(box);
        count++;
        break;
      }
    }
  }
}
