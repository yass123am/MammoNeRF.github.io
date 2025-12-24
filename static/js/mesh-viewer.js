document.addEventListener("DOMContentLoaded", () => {

  const container = document.getElementById("viewer");

  /* ======================
     SCENE
  ====================== */
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  /* ======================
     CAMERA
  ====================== */
  const camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.01,
    1e7
  );

  /* ======================
     RENDERER
  ====================== */
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  /* ======================
     CONTROLS
  ====================== */
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  /* ======================
     LIGHTING (MeshLab-like)
  ====================== */
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6));

  /* ======================
     LOAD PLY (NO MODIFICATION)
  ====================== */
  const loader = new THREE.PLYLoader();
  loader.load(
    "./model.ply",   // 🔁 CHANGE THIS PATH
    geometry => {

      // Only compute normals if missing
      if (!geometry.attributes.normal) {
        geometry.computeVertexNormals();
      }

      const material = geometry.attributes.color
        ? new THREE.MeshStandardMaterial({
            vertexColors: true,
            side: THREE.DoubleSide
          })
        : new THREE.MeshStandardMaterial({
            color: 0xaaaaaa,
            side: THREE.DoubleSide
          });

      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      /* ---- Camera framing ONLY (mesh untouched) ---- */
      geometry.computeBoundingBox();
      const box = geometry.boundingBox;

      const size = box.getSize(new THREE.Vector3()).length();
      const center = box.getCenter(new THREE.Vector3());

      const distance = size * 1.2;

      camera.position.copy(center).add(new THREE.Vector3(distance, distance, distance));
      camera.lookAt(center);

      controls.target.copy(center);
      controls.update();
    },
    undefined,
    error => {
      console.error("PLY load error:", error);
    }
  );

  /* ======================
     ANIMATION LOOP
  ====================== */
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  /* ======================
     RESIZE HANDLER
  ====================== */
  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

});
