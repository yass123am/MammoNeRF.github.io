document.addEventListener("DOMContentLoaded", function () {
  const viewers = [];
  const sceneGroups = [[], [], []];
  const pointMaterials = [];   // ⭐ store point materials
  window.renderers = [];
  let isUpdating = false;

  function init() {
    const containers = [];
    for (let i = 1; i <= 3; i++) {
      containers.push(document.getElementById(`mesh-container-${i}`));
    }

    const modelPaths = [
      './static/mesh/Volume_1.ply',
      './static/mesh/nerf_3d_grid.ply',
      './static/mesh/new_v.ply',
    ];

    containers.forEach((container, i) => {
      if (!container) return;

      const sceneIndex = Math.floor(i / 1);
      const viewerIndex = i % 1;

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

      const loader = new THREE.PLYLoader();
      loader.load(
        modelPaths[i],
        geometry => {
          geometry.computeBoundingBox();

          const center = geometry.boundingBox.getCenter(new THREE.Vector3());
          geometry.translate(-center.x, -center.y, -center.z);

          const size = geometry.boundingBox.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const targetSize = 1.5;
          const scale = targetSize / maxDim;

          const material = new THREE.PointsMaterial({
            size: 0.005,                // ⭐ default point size
            sizeAttenuation: true,
            vertexColors: !!geometry.attributes.color,
            color: geometry.attributes.color ? undefined : 0x3366cc
          });

          pointMaterials.push(material);   // ⭐ store material

          const points = new THREE.Points(geometry, material);
          points.scale.setScalar(scale);
          scene.add(points);

          const radius = 2.5;
          camera.position.set(radius, radius * 0.5, radius);
          camera.lookAt(0, 0, 0);
          controls.target.set(0, 0, 0);
          controls.update();
        },
        undefined,
        error => {
          console.error(`Failed to load ${modelPaths[i]}`, error);
        }
      );
    });

    setTimeout(setupCameraSync, 1000);
    animate();
  }

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

    const sceneViewers = sceneGroups[sceneIndex];
    const source = sceneViewers[sourceViewerIndex];
    if (!source) return;

    sceneViewers.forEach((viewer, index) => {
      if (index !== sourceViewerIndex) {
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
    viewers.forEach((viewer, i) => {
      const container = document.getElementById(`mesh-container-${i + 1}`);
      if (container && container.offsetParent !== null) {
        const w = container.clientWidth;
        const h = container.clientHeight;
        viewer.camera.aspect = w / h;
        viewer.camera.updateProjectionMatrix();
        viewer.renderer.setSize(w, h);
      }
    });
  });

  // ⭐ GLOBAL POINT SIZE CONTROL
  window.setPointSize = function (size) {
    pointMaterials.forEach(mat => {
      mat.size = size;
      mat.needsUpdate = true;
    });
  };

  init();
});
