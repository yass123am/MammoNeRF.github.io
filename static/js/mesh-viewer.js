document.addEventListener("DOMContentLoaded", function () {
  const viewers = [];
  const sceneGroups = [[], [], []]; // 3 examples
  window.renderers = [];
  let isUpdating = false;

  function init() {
    const containers = [];
    for (let i = 1; i <= 6; i++) {
      containers.push(document.getElementById(`mesh-container-${i}`));
    }

    // 3 examples × 2 PLY files
    const modelPaths = [
      './static/mesh/output_100mb.ply',
      './static/mesh/density_1.ply',

      './static/mesh/output_100mb_2.ply',
      './static/mesh/density_2.ply',

      './static/mesh/output_100mb_3.ply',
      './static/mesh/density_3.ply',
    ];

    containers.forEach((container, i) => {
      if (!container) return;

      // ---- indexing logic ----
      const sceneIndex = Math.floor(i / 2); // 0,0,1,1,2,2
      const viewerIndex = i % 2;            // 0 or 1
      const modelPath = modelPaths[i];

      // ---- THREE.js setup ----
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);

      const camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.001,
        1000
      );

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(container.clientWidth, container.clientHeight);
      container.appendChild(renderer.domElement);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;

      scene.add(new THREE.AmbientLight(0xffffff, 1.0));

      const viewer = {
        scene,
        camera,
        renderer,
        controls,
        container,
        sceneIndex,
        viewerIndex
      };

      viewers.push(viewer);
      sceneGroups[sceneIndex].push(viewer);
      window.renderers.push(renderer);

      // ---- Load PLY ----
      const loader = new THREE.PLYLoader();
      loader.load(
        modelPath,
        geometry => {
          geometry.computeBoundingBox();

          // Center
          const center = geometry.boundingBox.getCenter(new THREE.Vector3());
          geometry.translate(-center.x, -center.y, -center.z);

          // Scale
          const size = geometry.boundingBox.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 1.5 / maxDim;

          const material = new THREE.PointsMaterial({
            size: 0.01,
            sizeAttenuation: true,
            vertexColors: !!geometry.attributes.color,
            color: geometry.attributes.color ? undefined : 0x3366cc
          });

          const points = new THREE.Points(geometry, material);
          points.scale.setScalar(scale);
          scene.add(points);

          camera.position.set(2.5, 1.25, 2.5);
          camera.lookAt(0, 0, 0);
          controls.target.set(0, 0, 0);
          controls.update();
        },
        undefined,
        error => {
          console.error(`Failed to load ${modelPath}`, error);
        }
      );
    });

    setTimeout(setupCameraSync, 500);
    animate();
  }

  // ---- Camera sync within each example ----
  function setupCameraSync() {
    viewers.forEach(viewer => {
      viewer.controls.addEventListener('change', () => {
        if (!isUpdating) {
          syncCamerasInScene(viewer.sceneIndex, viewer.viewerIndex);
        }
      });
    });
  }

  function syncCamerasInScene(sceneIndex, sourceViewerIndex) {
    if (isUpdating) return;
    isUpdating = true;

    const group = sceneGroups[sceneIndex];
    const source = group[sourceViewerIndex];
    if (!source) {
      isUpdating = false;
      return;
    }

    group.forEach((viewer, idx) => {
      if (idx !== sourceViewerIndex) {
        viewer.camera.position.copy(source.camera.position);
        viewer.camera.quaternion.copy(source.camera.quaternion);
        viewer.controls.target.copy(source.controls.target);
        viewer.controls.update();
      }
    });

    isUpdating = false;
  }

  function animate() {
    requestAnimationFrame(animate);
    viewers.forEach(viewer => {
      if (viewer.container.offsetParent !== null) {
        viewer.controls.update();
        viewer.renderer.render(viewer.scene, viewer.camera);
      }
    });
  }

  window.addEventListener('resize', () => {
    viewers.forEach(viewer => {
      const { container, camera, renderer } = viewer;
      if (container && container.offsetParent !== null) {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      }
    });
  });

  init();
});
