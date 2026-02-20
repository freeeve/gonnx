package clip

import "github.com/advancedclimatesystems/gonnx/ops"

var clipVersions = ops.OperatorVersions{
	11: ops.NewOperatorConstructor(newClip, 11, clipTypeConstraints),
	13: ops.NewOperatorConstructor(newClip, 13, clipTypeConstraints),
}

func GetVersions() ops.OperatorVersions {
	return clipVersions
}
