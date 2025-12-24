document.addEventListener("DOMContentLoaded", () => {
  const viewers = [];
  const sceneGroups = [[], [], []];
  window.renderers = [];
  let isUpdating = false;

  const modelPaths = [
    './static/mesh/nerf_3d_grid.ply',
    './static/mesh/nerf_3d_grid.ply',
    './static/mesh/new_v.ply',
  ];

  function init() {
    const containers = Array.from({ length: 3 }, (_, i) =>
      document.getElementById(`mesh-container-${i + 1}`)
    );

    containers.forEach((container, i) => {
      if (!container) return;

      const sceneIndex = Math.floor(i / 1);

      const scene = createScene();
      const camera = createCamera(container);
      const renderer = createRenderer(container);
      const controls = createControls(camera, renderer);

      const viewer = { scene, camera, renderer, controls, container, sceneIndex, viewerIndex: 0 };
      viewers.push(viewer);
      sceneGroups[sceneIndex].push(viewer);
      window.renderers.push(renderer);

      loadPLY(modelPaths[i], scene, camera, controls);
    });

    setTimeout(setupCameraSync, 1000);
    animate();
  }

  function createScene() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    scene.add(new THREE.DirectionalLight(0xffffff, 0.4).position.set(5, 5, 5));
    scene.add(new THREE.DirectionalLight(0xffffff, 0.4).position.set(-5, 5, -5));
    return scene;
  }

  function createCamera(container) {
    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.01,
      100
    );
    camera.position.set(0, 0, 2);
    return camera;
  }

  function createRenderer(container) {
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    return renderer;
  }

  function createControls(camera, renderer) {
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    return controls;
  }

  function loadPLY(path, scene, camera, controls) {
    const loader = new THREE.PLYLoader();
    loader.load(
      path,
      geometry => {
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();

        const center = geometry.boundingBox.getCenter(new THREE.Vector3());
        geometry.translate(-center.x, -center.y, -center.z);

        const size = geometry.boundingBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const targetSize = 1.5;
        const scale = targetSize / maxDim;

        const material = geometry.attributes.color
          ? new THREE.MeshLambertMaterial({ vertexColors: true })
          : new THREE.MeshLambertMaterial({ color: 0x8888aa });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.scale.setScalar(scale);
        scene.add(mesh);

        const radius = 2.5;
        camera.position.set(radius, radius * 0.5, radius);
        camera.lookAt(0, 0, 0);
        controls.target.set(0, 0, 0);
        controls.update();
      },
      undefined,
      error => {
        console.error(`Failed to load mesh ${path}`, error);
        const fallback = new THREE.Mesh(
          new THREE.BoxGeometry(1, 1, 1),
          new THREE.MeshLambertMaterial({ color: 0xff6b6b })
        );
        scene.add(fallback);
      }
    );
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
        viewer.camera.rotation.copy(source.camera.rotation);
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
        const width = container.clientWidth;
        const height = container.clientHeight;
        viewer.camera.aspect = width / height;
        viewer.camera.updateProjectionMatrix();
        viewer.renderer.setSize(width, height);
      }
    });
  });

  init();
});
