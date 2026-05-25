extends CharacterBody3D

## HD-2D directional mesh-swap controller for a CharacterBody3D.
##
## Replaces an AnimatedSprite3D billboard by swapping the Mesh on a single
## MeshInstance3D child ("CharacterMesh") to display the correct pre-built
## .glb frame based on movement direction and animation state.
##
## Expected GLB layout (mirrors blender_mesh_builder.py output):
##   res://assets/glb_files/{animation}_mesh/directions/{direction}/{direction}{frame}.glb
##
## Frame numbering in the actual files:
##   idle  →  {dir}0.glb          (one frame per direction, index 0)
##   run   →  {dir}1.glb … {dir}N.glb  (N frames per direction, starting at 1)
##
## The cache builder probes for the real start index at runtime, so a future
## re-export that includes a run frame 0 will still work without code changes.
##
## ── Input Map actions required ─────────────────────────────────────────────
##   move_left, move_right, move_forward, move_back, jump
##   (add these in Project → Project Settings → Input Map)
##
## ── Coordinate convention ──────────────────────────────────────────────────
##   World X+  = East     World X−  = West
##   World Z−  = North    World Z+  = South   (standard Godot 3D / glTF Y-up)
##   "forward" input  →  Z−  (north),   "back" input  →  Z+  (south)
##   Adjust the _vel_to_direction() atan2 arguments if your camera or input
##   remapping uses a different convention.


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

## Horizontal movement speed (m/s).
@export var speed              : float = 5.0
## Vertical impulse applied on jump.
@export var jump_velocity      : float = 4.5
## World-space height above the character's origin that the camera looks at.
@export var camera_look_height : float = 1.0

## Run-cycle playback rate.
const ANIM_FPS : float = 8.0

## Minimum horizontal speed (m/s) required to switch to the run animation.
const DIR_THRESHOLD : float = 0.1

## World-space offset from the character's origin to the camera position.
## X=0 centres the camera,  Y=4 lifts it above,  Z=6 pulls it behind.
const CAMERA_OFFSET : Vector3 = Vector3(0, 4, 6)

## Base path for all GLB mesh assets inside the Godot project.
const GLB_BASE : String = "res://assets/glb_files/"

## Compass names, clockwise from North, matching blender_mesh_builder.py.
const DIRECTIONS : Array[String] = [
	"north", "north_east", "east", "south_east",
	"south", "south_west", "west", "north_west",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal state
# ─────────────────────────────────────────────────────────────────────────────

## mesh_cache["idle"|"run"]["north"|…] → Array of {mesh:Mesh, mat:Material} Dictionaries.
var mesh_cache    : Dictionary = {}
var cur_direction : String     = "south"
var cur_animation : String     = "idle"
var frame_index   : int        = 0
var frame_timer   : float      = 0.0

@onready var character_mesh : MeshInstance3D = get_node_or_null("CharacterMesh") as MeshInstance3D
@onready var camera         : Camera3D        = get_node_or_null("Camera3D")     as Camera3D


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

func _ready() -> void:
	if not character_mesh:
		push_error("HD2D: 'CharacterMesh' must be a MeshInstance3D child of this node. " +
				   "Right-click CharacterBody3D → Add Child Node → MeshInstance3D, name it CharacterMesh.")
		return
	if not camera:
		push_warning("HD2D: no Camera3D child named 'Camera3D' found — camera follow disabled. " +
					 "Add Child Node → Camera3D under CharacterBody3D and name it Camera3D.")
	_build_mesh_cache()
	_apply_mesh()
	if camera:
		camera.position = CAMERA_OFFSET


func _physics_process(delta: float) -> void:
	_apply_gravity(delta)
	_apply_movement()
	move_and_slide()
	_tick_animation(delta)
	_apply_camera()


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

## Rotates the camera to look at the character's mid-point each physics frame.
## Position is set once in _ready() via camera.position = CAMERA_OFFSET and
## then follows automatically because the camera is a child node.
func _apply_camera() -> void:
	if camera:
		camera.look_at(global_position + Vector3(0, camera_look_height, 0), Vector3.UP)


# ─────────────────────────────────────────────────────────────────────────────
# Cache building
# ─────────────────────────────────────────────────────────────────────────────

func _build_mesh_cache() -> void:
	print("HD2D: building mesh cache…")
	var total := 0

	for anim : String in ["idle", "run"]:
		mesh_cache[anim] = {}
		for dir : String in DIRECTIONS:
			var frames : Array = _load_frames(anim, dir)
			mesh_cache[anim][dir] = frames
			total += frames.size()
			if frames.is_empty():
				push_warning(
					"HD2D: no GLB frames found for [%s][%s] — check res://assets/glb_files/"
					% [anim, dir]
				)

	print("HD2D: cache ready — %d mesh frames loaded." % total)


## Discovers and loads all frames for one animation + direction combination.
## Probes for the real start index (0 or 1) so the function works regardless
## of whether the exporter wrote a frame 0 for that animation.
func _load_frames(anim: String, dir: String) -> Array:
	var frames : Array = []

	# Find the first index that actually exists on disk (0 or 1).
	var start := 0
	if not ResourceLoader.exists(_glb_path(anim, dir, 0)):
		start = 1

	var f := start
	while true:
		var path := _glb_path(anim, dir, f)
		if not ResourceLoader.exists(path):
			break       # frames are sequential; stop at the first gap
		var data := _extract_mesh_from_glb(path)
		if not data.is_empty():
			frames.append(data)
		f += 1

	return frames


func _glb_path(anim: String, dir: String, frame: int) -> String:
	return "%s%s_mesh/directions/%s/%s%d.glb" % [GLB_BASE, anim, dir, dir, frame]


## Instantiates the PackedScene at path, finds the first MeshInstance3D, extracts
## its Mesh resource AND its surface-0 material, then frees the temporary node tree.
## Both are returned as a Dictionary so _apply_mesh() can re-apply the correct
## material on every swap (preventing stale overrides from bleeding across frames).
func _extract_mesh_from_glb(path: String) -> Dictionary:
	var scene := load(path) as PackedScene
	if not scene:
		push_warning("HD2D: could not load PackedScene from '%s'" % path)
		return {}
	var root  := scene.instantiate()
	var mi    := _find_mesh_instance_recursive(root)
	var result : Dictionary = {}
	if mi and mi.mesh:
		# Prefer the material embedded in the mesh surface; fall back to the
		# MeshInstance3D's surface override if the surface slot is unset.
		var mat : Material = mi.mesh.surface_get_material(0)
		if mat == null:
			mat = mi.get_surface_override_material(0)
		result = {"mesh": mi.mesh, "mat": mat}
	root.free()
	if result.is_empty():
		push_warning("HD2D: no MeshInstance3D found inside '%s'" % path)
	return result


func _find_mesh_instance_recursive(node: Node) -> MeshInstance3D:
	if node is MeshInstance3D:
		return node as MeshInstance3D
	for child : Node in node.get_children():
		var mi := _find_mesh_instance_recursive(child)
		if mi != null:
			return mi
	return null


# ─────────────────────────────────────────────────────────────────────────────
# Movement
# ─────────────────────────────────────────────────────────────────────────────

func _apply_gravity(delta: float) -> void:
	if not is_on_floor():
		velocity += get_gravity() * delta


func _apply_movement() -> void:
	# get_vector returns (left/right, forward/back) normalised to the unit circle.
	# input.x → world X (+east / −west)
	# input.y → world Z (+south / −north, because "forward" = north = −Z)
	var raw := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	if raw != Vector2.ZERO:
		velocity.x = raw.x * speed
		velocity.z = raw.y * speed
	else:
		velocity.x = move_toward(velocity.x, 0.0, speed)
		velocity.z = move_toward(velocity.z, 0.0, speed)

	if Input.is_action_just_pressed("jump") and is_on_floor():
		velocity.y = jump_velocity


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────

func _tick_animation(delta: float) -> void:
	var horiz     := Vector2(velocity.x, velocity.z)
	var is_moving := horiz.length() > DIR_THRESHOLD

	var new_anim := "run" if is_moving else "idle"
	# When the character stops, keep the last direction so the idle mesh
	# matches the pose it was in when it came to a halt.
	var new_dir  := _vel_to_direction(horiz) if is_moving else cur_direction

	var dirty := false

	# Animation type changed (idle ↔ run): restart from frame 0.
	if new_anim != cur_animation:
		cur_animation = new_anim
		frame_index   = 0
		frame_timer   = 0.0
		dirty         = true

	# Direction changed within the same animation: swap the mesh but do NOT
	# reset frame_index — this avoids a visual pop during a run-cycle turn.
	if new_dir != cur_direction:
		cur_direction = new_dir
		dirty         = true

	# Advance the run cycle on the per-frame timer.
	if cur_animation == "run":
		frame_timer += delta
		var spf := 1.0 / ANIM_FPS
		while frame_timer >= spf:
			frame_timer -= spf
			var count := _frame_count(cur_animation, cur_direction)
			if count > 0:
				frame_index = (frame_index + 1) % count
				dirty = true

	if dirty:
		_apply_mesh()


## Maps a 2-D velocity vector to one of the eight DIRECTIONS strings.
## vel.x = world X (+east),  vel.y = world Z (+south / −north).
## atan2(x, −y) gives a clockwise angle measured from North (the −Z axis).
func _vel_to_direction(vel: Vector2) -> String:
	var angle  := atan2(vel.x, -vel.y)
	# Offset by half a sector (PI/8) so each name's range is centred on its
	# exact compass angle, then divide into eight 45-degree (PI/4) buckets.
	var sector := int(fposmod(angle + PI / 8.0, TAU) / (PI / 4.0))
	return DIRECTIONS[sector % 8]


func _frame_count(anim: String, dir: String) -> int:
	var anim_dict = mesh_cache.get(anim)
	if anim_dict == null:
		return 0
	var frames = anim_dict.get(dir)
	if frames == null:
		return 0
	return (frames as Array).size()


# ─────────────────────────────────────────────────────────────────────────────
# Mesh swap
# ─────────────────────────────────────────────────────────────────────────────

func _apply_mesh() -> void:
	var anim_dict = mesh_cache.get(cur_animation)
	if anim_dict == null:
		return
	var frames = (anim_dict as Dictionary).get(cur_direction)
	if frames == null or (frames as Array).is_empty():
		return
	var arr  : Array      = frames
	var data : Dictionary = arr[frame_index % arr.size()]
	var mesh : Mesh       = data.get("mesh")
	if mesh == null:
		return
	# Always re-apply both mesh and its surface-0 material together.
	# This prevents a stale material override left on the CharacterMesh node
	# from a previous direction (or editor assignment) bleeding onto the new
	# frame — the root cause of the north-direction texture displacement bug.
	if character_mesh.mesh != mesh:
		character_mesh.mesh = mesh
	var mat : Material = data.get("mat")
	character_mesh.set_surface_override_material(0, mat)
